
version = "0.0.2-SNAPSHOT"

buildscript {
    repositories {
        jcenter()
    }
    dependencies {
        classpath ("com.github.jengelman.gradle.plugins:shadow:5.0.0")
    }
}


allprojects {
    apply(plugin = "java")
    apply(plugin="com.github.johnrengelman.shadow")

    repositories {
        jcenter()
    }

    val implementation by configurations
    val compileOnly by configurations
    dependencies {
        implementation("commons-cli:commons-cli:1.2")
        implementation("log4j:log4j:1.2.17")
        compileOnly("org.apache.spark:spark-core_2.11:2.3.2")
        compileOnly("org.apache.spark:spark-sql_2.11:2.3.2")
        compileOnly("org.apache.spark:spark-hive_2.11:2.3.2")
        implementation("com.google.protobuf:protobuf-java:3.7.1")
    }

    configure<JavaPluginExtension> {
        sourceCompatibility = JavaVersion.VERSION_1_8
    }

    tasks.withType<JavaCompile> {
        options.encoding = "UTF-8"
    }
}
