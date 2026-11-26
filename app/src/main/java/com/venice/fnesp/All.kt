package com.venice.fnesp

import ai.onnxruntime.*
import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.*
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.*
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("fnesp", Context.MODE_PRIVATE)
    var confidence: Float
        get() = sp.getFloat("conf", 0.55f); set(v) = sp.edit().putFloat("conf", v).apply()
    var alpha: Int
        get() = sp.getInt("alpha", 90); set(v) = sp.edit().putInt("alpha", v).apply()
    var roiFraction: Float
        get() = sp.getFloat("roi", 0.45f); set(v) = sp.edit().putFloat("roi", v).apply()
    var boxThickness: Float
        get() = sp.getFloat("thick", 2.5f); set(v) = sp.edit().putFloat("thick", v).apply()
    var frameSkip: Int
        get() = sp.getInt("skip", 2); set(v) = sp.edit().putInt("skip", v).apply()
    var playerOnly: Boolean
        get() = sp.getBoolean("playeronly", true); set(v) = sp.edit().putBoolean("playeronly", v).apply()
    var chams: Boolean
        get() = sp.getBoolean("chams", true); set(v) = sp.edit().putBoolean("chams", v).apply()
    var captureSize: Int
        get() = sp.getInt("capture", 480); set(v) = sp.edit().putInt("capture", v).apply()
}

data class Raw(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float)

class Detector(modelPath: String, private val inputSize: Int = 320) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inName: String
    private val canvasBmp = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBmp)
    private val mat = Matrix()
    private val buf = ByteBuffer.allocateDirect(3 * inputSize * inputSize * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer()

    init {
        val o = OrtSession.SessionOptions()
        o.setIntraOpNumThreads(2)
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelPath, o)
        inName = session.inputNames.first()
    }

    fun detect(src: Bitmap, conf: Float, playerOnly: Boolean): List<Raw> {
        val s = inputSize.toFloat()
        val scale = min(s / src.width, s / src.height)
        val nw = (src.width * scale).toInt()
        val nh = (src.height * scale).toInt()
        val dx = (inputSize - nw) / 2
        val dy = (inputSize - nh) / 2

        canvas.drawColor(Color.rgb(114, 114, 114))
        mat.reset(); mat.setScale(scale, scale); mat.postTranslate(dx.toFloat(), dy.toFloat())
        canvas.drawBitmap(src, mat, null)

        val px = IntArray(inputSize * inputSize)
        canvasBmp.getPixels(px, 0, inputSize, 0, 0, inputSize, inputSize)

        buf.clear()
        for (c in 0..2) for (i in px.indices) {
            val p = px[i]
            val v = when (c) {
                0 -> (p shr 16) and 0xFF
                1 -> (p shr 8) and 0xFF
                else -> p and 0xFF
            }
            buf.put(v / 255f)
        }
        buf.rewind()

        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val out = session.run(mapOf(inName to OnnxTensor.createTensor(env, buf, shape))).use { r ->
            val t = r[0] as OnnxTensor
            val dims = t.info.shape
            val ch = dims[1].toInt()
            val an = dims[2].toInt()
            val fb = t.floatBuffer
            val arr = FloatArray(fb.remaining()); fb.get(arr)
            return@use Triple(arr, ch, an)
        }

        val data = out.first
        val ch = out.second
        val an = out.third
        val nc = ch - 4
        val res = ArrayList<Raw>(32)
        for (i in 0 until an) {
            var best = 0f
            var bestC = -1
            if (playerOnly) {
                best = data[4 * an + i]; bestC = 0
            } else {
                for (c in 0 until nc) {
                    val v = data[(4 + c) * an + i]
                    if (v > best) { best = v; bestC = c }
                }
            }
            if (best < conf || bestC < 0) continue
            val cx = data[i]; val cy = data[an + i]
            val w = data[2 * an + i]; val h = data[3 * an + i]
            val x1 = (cx - w / 2 - dx) / scale
            val y1 = (cy - h / 2 - dy) / scale
            val x2 = (cx + w / 2 - dx) / scale
            val y2 = (cy + h / 2 - dy) / scale
            if (x2 <= 0 || y2 <= 0 || x1 >= src.width || y1 >= src.height) continue
            val bw = max(1f, x2 - x1); val bh = max(1f, y2 - y1)
            if (bh / bw < 1.05f) continue
            val area = bw * bh / (src.width * src.height)
            if (area < 0.0006f || area > 0.45f) continue
            res.add(Raw(x1, y1, x2, y2, best))
        }
        return nms(res, 0.45f)
    }

    private fun nms(list: List<Raw>, thr: Float): List<Raw> {
        val sorted = list.sortedByDescending { it.score }.toMutableList()
        val keep = ArrayList<Raw>(sorted.size)
        while (sorted.isNotEmpty()) {
            val a = sorted.removeAt(0)
            keep.add(a)
            sorted.removeAll { b -> iou(a, b) > thr }
        }
        return keep
    }

    private fun iou(a: Raw, b: Raw): Float {
        val x1 = max(a.x1, b.x1); val y1 = max(a.y1, b.y1)
        val x2 = min(a.x2, b.x2); val y2 = min(a.y2, b.y2)
        val iw = max(0f, x2 - x1); val ih = max(0f, y2 - y1)
        val inter = iw * ih
        val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (ua <= 0f) 0f else inter / ua
    }
}

class Track(val id: Int, var r: Raw, var life: Int = 8)

class Tracker {
    private var next = 1
    private val tracks = ArrayList<Track>()

    fun update(raws: List<Raw>): List<Track> {
        val used = BooleanArray(raws.size)
        for (t in tracks) {
            var bi = -1; var bs = 0.25f
            for (i in raws.indices) {
                if (used[i]) continue
                val s = iou(t.r, raws[i])
                if (s > bs) { bs = s; bi = i }
            }
            if (bi >= 0) {
                used[bi] = true
                t.r = raws[bi]
                t.life = 8
            } else t.life--
        }
        for (i in raws.indices) if (!used[i]) tracks.add(Track(next++, raws[i]))
        tracks.removeAll { it.life <= 0 }
        return tracks
    }

    fun clear() = tracks.clear()

    private fun iou(a: Raw, b: Raw): Float {
        val x1 = maxOf(a.x1, b.x1); val y1 = maxOf(a.y1, b.y1)
        val x2 = minOf(a.x2, b.x2); val y2 = minOf(a.y2, b.y2)
        val iw = maxOf(0f, x2 - x1); val ih = maxOf(0f, y2 - y1)
        val inter = iw * ih
        val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (ua <= 0f) 0f else inter / ua
    }
}

class OverlayView(ctx: Context) : View(ctx) {
    var tracks: List<Track> = emptyList()
    var roi: RectF = RectF()
    var roiSize: Float = 1f
    var alpha = 90
    var thickness = 2.5f
    var chams = true
    var fps = 0f
    var paused = false

    private val box = Paint().apply { style = Paint.Style.STROKE; isAntiAlias = true }
    private val fill = Paint().apply { style = Paint.Style.FILL }
    private val text = Paint().apply { color = Color.WHITE; textSize = 26f; isAntiAlias = true }

    override fun onDraw(c: Canvas) {
        if (paused) return
        val sx = roi.width() / roiSize
        val sy = roi.height() / roiSize
        for (t in tracks) {
            val r = t.r
            val l = roi.left + r.x1 * sx
            val tp = roi.top + r.y1 * sy
            val rr = roi.left + r.x2 * sx
            val b = roi.top + r.y2 * sy
            val fresh = t.life >= 7
            val col = if (fresh) Color.rgb(0, 230, 118) else Color.rgb(255, 179, 0)
            if (chams) {
                fill.color = Color.argb((alpha * if (fresh) 1f else 0.55f).toInt(),
                    Color.red(col), Color.green(col), Color.blue(col))
                c.drawRect(l, tp, rr, b, fill)
            }
            box.color = col
            box.strokeWidth = thickness
            c.drawRect(l, tp, rr, b, box)
            c.drawText((t.r.score * 100f).toInt().toString() + "%", l, tp - 6f, text)
        }
        c.drawText(fps.toInt().toString() + " fps · " + tracks.size + " trk", 20f, height - 30f, text)
    }
}

class OverlayService : Service() {
    private lateinit var wm: WindowManager
    private lateinit var prefs: Prefs
    private var view: OverlayView? = null
    private var ball: View? = null
    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var detector: Detector? = null
    private val tracker = Tracker()
    private var running = false
    private var paused = false
    private var frame = 0
    private var lastT = 0L
    private var fps = 0f
    private var capW = 0
    private var capH = 0
    private var roiRect = RectF()
    private val main = Handler(Looper.getMainLooper())
    private val worker = Handler(Looper.getMainLooper())

    override fun onBind(i: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        wm = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        prefs = Prefs(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == "stop") { stopSelf(); return START_NOT_STICKY }
        if (running) return START_STICKY

        val notif = Notification.Builder(this, chan())
            .setContentTitle("FN ESP")
            .setContentText("Detection running")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setOngoing(true).build()
        startForeground(1, notif)

        val data = intent?.getParcelableExtra<Intent>("data") ?: return START_NOT_STICKY
        val mgr = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        projection = mgr.getMediaProjection(intent.getIntExtra("code", -1), data)

        val dm = resources.displayMetrics
        capH = prefs.captureSize
        capW = (capH * dm.widthPixels.toFloat() / dm.heightPixels).toInt()

        try {
            detector = Detector(copyModel(), if (prefs.captureSize >= 640) 640 else 320)
        } catch (e: Exception) {
            toast("Model missing or failed: " + e.message)
            stopSelf()
            return START_NOT_STICKY
        }

        addOverlay(dm.widthPixels, dm.heightPixels)

        reader = ImageReader.newInstance(capW, capH, PixelFormat.RGBA_8888, 2)
        display = projection?.createVirtualDisplay("cap", capW, capH, dm.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, reader?.surface, null, worker)

        running = true
        worker.post(loop)
        return START_STICKY
    }

    private val loop = object : Runnable {
        override fun run() {
            if (!running) return
            frame++
            if (!paused && frame % prefs.frameSkip == 0) tick()
            worker.postDelayed(this, 16)
        }
    }

    private fun tick() {
        val img = reader?.acquireLatestImage() ?: return
        try {
            val side = (min(capW, capH) * prefs.roiFraction).toInt().coerceAtLeast(160)
            val left = (capW - side) / 2
            val top = (capH - side) / 2
            val bmp = Bitmap.createBitmap(capW, capH, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(img.planes[0].buffer)
            val roi = Bitmap.createBitmap(bmp, left, top, side, side)
            bmp.recycle()

            val raws = detector?.detect(roi, prefs.confidence, prefs.playerOnly) ?: emptyList()
            roi.recycle()

            val tr = tracker.update(raws)

            val dm = resources.displayMetrics
            val vSide = min(dm.widthPixels, dm.heightPixels) * prefs.roiFraction
            val cx = dm.widthPixels / 2f
            val cy = dm.heightPixels / 2f
            roiRect.set(cx - vSide / 2, cy - vSide / 2, cx + vSide / 2, cy + vSide / 2)

            val now = System.currentTimeMillis()
            if (lastT > 0) fps = fps * 0.8f + (1000f / (now - lastT)) * 0.2f
            lastT = now

            main.post {
                view?.apply {
                    tracks = tr
                    roi = roiRect
                    roiSize = side.toFloat()
                    alpha = prefs.alpha
                    thickness = prefs.boxThickness
                    chams = prefs.chams
                    this.fps = fps
                    paused = paused
                    invalidate()
                }
            }
        } catch (e: Exception) {
        } finally {
            img.close()
        }
    }

    private fun addOverlay(w: Int, h: Int) {
        val v = OverlayView(this)
        val lp = WindowManager.LayoutParams(w, h,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT)
        lp.gravity = Gravity.TOP or Gravity.START
        wm.addView(v, lp)
        view = v

        val b = View(this)
        b.setBackgroundColor(0x8800C853.toInt())
        val blp = WindowManager.LayoutParams(70, 70,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE, PixelFormat.TRANSLUCENT)
        blp.gravity = Gravity.TOP or Gravity.END
        blp.x = 24
        blp.y = 24
        b.setOnClickListener {
            paused = !paused
            b.setBackgroundColor(if (paused) 0x88FFB300.toInt() else 0x8800C853.toInt())
        }
        b.setOnLongClickListener { stopSelf(); true }
        wm.addView(b, blp)
        ball = b
    }

    private fun copyModel(): String {
        val f = File(filesDir, "model.onnx")
        if (!f.exists()) assets.open("yolov8n.onnx").use { i -> f.outputStream().use { o -> i.copyTo(o) } }
        return f.absolutePath
    }

    private fun chan(): String {
        val c = NotificationChannel("fnesp", "FN ESP", NotificationManager.IMPORTANCE_LOW)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(c)
        return "fnesp"
    }

    private fun toast(s: String) {
        main.post { Toast.makeText(this, s, Toast.LENGTH_LONG).show() }
    }

    override fun onDestroy() {
        running = false
        worker.removeCallbacksAndMessages(null)
        display?.release()
        reader?.close()
        projection?.stop()
        view?.let { runCatching { wm.removeView(it) } }
        ball?.let { runCatching { wm.removeView(it) } }
        super.onDestroy()
    }
}

class MainActivity : Activity() {
    private lateinit var prefs: Prefs
    private val REQ = 1001

    override fun onCreate(s: Bundle?) {
        super.onCreate(s)
        prefs = Prefs(this)
        val sc = ScrollView(this)
        val ll = LinearLayout(this)
        ll.orientation = LinearLayout.VERTICAL
        ll.setPadding(48, 48, 48, 48)
        sc.addView(ll)

        fun head(t: String) {
            ll.addView(TextView(this).apply {
                text = t
                textSize = 16f
                setTextColor(Color.parseColor("#10606F"))
                setPadding(0, 32, 0, 8)
            })
        }

        fun slider(label: String, from: Int, to: Int, cur: Int, set: (Int) -> Unit) {
            head("$label — $cur")
            val sb = SeekBar(this)
            sb.max = to - from
            sb.progress = cur - from
            sb.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(x: SeekBar?, p: Int, f: Boolean) {
                    val v = p + from
                    set(v)
                    (ll.getChildAt(ll.childCount - 2) as? TextView)?.text = "$label — $v"
                }
                override fun onStartTrackingTouch(x: SeekBar?) {}
                override fun onStopTrackingTouch(x: SeekBar?) {}
            })
            ll.addView(sb)
        }

        head("FN ESP — box + chams overlay")
        slider("Confidence %", 25, 90, (prefs.confidence * 100).toInt()) { prefs.confidence = it / 100f }
        slider("Chams opacity %", 0, 100, prefs.alpha) { prefs.alpha = it }
        slider("Box thickness x10", 10, 60, (prefs.boxThickness * 10).toInt()) { prefs.boxThickness = it / 10f }
        slider("ROI size %", 20, 90, (prefs.roiFraction * 100).toInt()) { prefs.roiFraction = it / 100f }
        slider("Frame skip", 1, 4, prefs.frameSkip) { prefs.frameSkip = it }
        slider("Capture height", 320, 640, prefs.captureSize) { prefs.captureSize = it }

        val chams = CheckBox(this)
        chams.text = "Chams fill"
        chams.isChecked = prefs.chams
        chams.setOnCheckedChangeListener { _, b -> prefs.chams = b }
        ll.addView(chams)

        val po = CheckBox(this)
        po.text = "Model is player-only (custom)"
        po.isChecked = prefs.playerOnly
        po.setOnCheckedChangeListener { _, b -> prefs.playerOnly = b }
        ll.addView(po)

        val start = Button(this)
        start.text = "Grant overlay + start"
        start.setOnClickListener {
            if (!Settings.canDrawOverlays(this@MainActivity)) {
                startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")))
                return@setOnClickListener
            }
            val m = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
            startActivityForResult(m.createScreenCaptureIntent(), REQ)
        }
        ll.addView(start)

        val stop = Button(this)
        stop.text = "Stop"
        stop.setOnClickListener {
            startService(Intent(this@MainActivity, OverlayService::class.java).setAction("stop"))
        }
        ll.addView(stop)

        ll.addView(TextView(this).apply {
            text = "Grant overlay, then start, and pick Entire screen."
            setTextColor(Color.parseColor("#565E6B"))
            setPadding(0, 32, 0, 0)
        })

        setContentView(sc)
    }

    override fun onActivityResult(r: Int, c: Int, d: Intent?) {
        super.onActivityResult(r, c, d)
        if (r == REQ && c == RESULT_OK && d != null) {
            startForegroundService(Intent(this, OverlayService::class.java)
                .putExtra("data", d).putExtra("code", c))
        }
    }
}
