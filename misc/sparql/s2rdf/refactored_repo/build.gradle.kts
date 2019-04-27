
buildscript {
    repositories {
        jcenter()
    }
    dependencies {
        classpath ("com.github.jengelman.gradle.plugins:shadow:5.0.0")
    }
}

subprojects {
    apply(plugin="java")
    apply(plugin="com.github.johnrengelman.shadow")

    repositories {
        jcenter()
    }

    val compileOnly by configurations
    dependencies {
        compileOnly("org.apache.spark:spark-core_2.11:2.3.2")
        compileOnly("org.apache.spark:spark-sql_2.11:2.3.2")
    }
}
