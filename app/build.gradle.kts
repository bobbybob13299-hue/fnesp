plugins { id("com.android.application") }

android {
    namespace = "com.venice.fnesp"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.venice.fnesp"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildTypes { release { isMinifyEnabled = false } }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
