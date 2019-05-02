

tasks.withType<Jar>{
    manifest {
        attributes("Main-Class" to "queryTranslator.run.Main")
    }
}

val implementation by configurations

dependencies {
    implementation("org.apache.jena:jena-core:3.10.0")
    implementation("org.apache.jena:jena-iri:3.10.0")
    implementation("org.apache.jena:jena-arq:3.10.0")
    implementation("commons-cli:commons-cli:1.4")
    implementation("xerces:xercesImpl:2.12.0")
    implementation("log4j:log4j:1.2.17")
    implementation("org.slf4j:slf4j-api:1.6.4")
    implementation("org.slf4j:slf4j-log4j12:1.6.4")
    implementation("xml-apis:xml-apis:1.4.01")
}
