
plugins {
    "maven-publish"
}

val implementation by configurations
val testImplementation by configurations
dependencies {
    implementation("junit:junit:4.12")
    implementation("org.junit.jupiter:junit-jupiter-api:5.4.1")
    testImplementation("com.holdenkarau:spark-testing-base_2.11:2.2.0_0.10.0")
}

group = "prost"
description = "prost-loader"

/*
publishing {
    publications {
        maven(MavenPublication) {
            from(components.java)
        }
    }
}
*/

