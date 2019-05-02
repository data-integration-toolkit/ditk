package run;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.cli.*;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import executor.Executor;
import joinTree.JoinTree;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import translator.Stats;
import translator.Translator;

/**
 * The Main class parses the CLI arguments and calls the translator and the executor.
 *
 * @author Matteo Cossu
 */
public class Main {

	private static final Logger logger = Logger.getLogger("PRoST");

	private static String inputFile;
	// CSCI-548
	private static String inputQuery;
	private static String outputFile;
	private static String statsFileName = "";
	private static String databasePath;
	private static int treeWidth = -1;
	// private static boolean useVP = false;
	private static boolean usePropertyTable = false;
	private static boolean useInversePropertyTable = false;
	private static boolean useJoinedPropertyTable = false;
	private static int setGroupSize = -1;
	private static boolean benchmarkMode = false;
	private static String benchmark_file;
	private static String loj4jFileName = "log4j.properties";

	public static void main(final String[] args) throws IOException {
		final InputStream inStream = Main.class.getClassLoader().getResourceAsStream(loj4jFileName);
		final Properties props = new Properties();
		props.load(inStream);
		PropertyConfigurator.configure(props);

		/*
		 * Manage the CLI options
		 */
		final CommandLineParser parser = new PosixParser();
		final Options options = new Options();

		// CSCI- 548
		final Option queryOption = new Option("q", "query", true, "Comma-seperated SPARQL Query");
		queryOption.setRequired(true);
		options.addOption(queryOption);

		final Option inputOpt = new Option("i", "input", true, "Input file with the SPARQL query.");
		//inputOpt.setRequired(true);

		options.addOption(inputOpt);
		final Option outputOpt = new Option("o", "output", true, "Path for the results in HDFS.");
		options.addOption(outputOpt);
		final Option statOpt = new Option("s", "stats", true, "File with statistics (required)");
		options.addOption(statOpt);
		statOpt.setRequired(true);
		final Option databaseOpt = new Option("d", "DB", true, "Database containing the VP tables and the PT.");
		databaseOpt.setRequired(true);
		options.addOption(databaseOpt);
		final Option helpOpt = new Option("h", "help", true, "Print this help.");
		options.addOption(helpOpt);
		final Option widthOpt = new Option("w", "width", true, "The maximum Tree width");
		options.addOption(widthOpt);
		final Option propertyTableOpt = new Option("wpt", "property_table", false, "Use Wide Propery Table");
		options.addOption(propertyTableOpt);
		final Option reversePropertyTableOpt =
				new Option("iwpt", "inverse_property_table", false, "Use Inverse Wide Property Table");
		options.addOption(reversePropertyTableOpt);
		final Option joinedPropertyTable =
				new Option("jwpt", "joined_property_table", false, "Use Joined Wide Property Table");
		options.addOption(joinedPropertyTable);
		final Option benchmarkOpt = new Option("t", "times", true, "Save the time results in a csv file.");
		options.addOption(benchmarkOpt);
		final Option groupsizeOpt =
				new Option("g", "groupsize", true, "Minimum Group Size for Wide Property Table nodes");
		options.addOption(groupsizeOpt);

		final HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (final MissingOptionException e) {
			formatter.printHelp("JAR", "Execute a  SPARQL query with Spark", options, "", true);
			return;
		} catch (final ParseException e) {
			e.printStackTrace();
		}

		if (cmd.hasOption("help")) {
			formatter.printHelp("JAR", "Execute a  SPARQL query with Spark", options, "", true);
			return;
		}
		if (cmd.hasOption("input")) {
			inputFile = cmd.getOptionValue("input");
		}
		if(cmd.hasOption("query")) {
			inputQuery = cmd.getOptionValue("query");
		}
		if(cmd.hasOption("query")) {
			inputQuery = cmd.getOptionValue("query");
		}

		if (cmd.hasOption("output")) {
			outputFile = cmd.getOptionValue("output");
			logger.info("Output file set to:" + outputFile);
		}
		if (cmd.hasOption("stats")) {
			statsFileName = cmd.getOptionValue("stats");
		}
		if (cmd.hasOption("width")) {
			treeWidth = Integer.valueOf(cmd.getOptionValue("width"));
			logger.info("Maximum tree width is set to " + String.valueOf(treeWidth));
		}
		// if (cmd.hasOption("only_vp")) {
		// useVP = true;
		// logger.info("Using Vertical Partitioning only.");
		// }
		if (cmd.hasOption("property_table")) {
			usePropertyTable = true;
			logger.info("Using Wide Property Table.");
		}
		if (cmd.hasOption("inverse_property_table")) {
			useInversePropertyTable = true;
			logger.info("Using Inverse Wide Property Table.");
		}
		if (cmd.hasOption("joined_property_table")) {
			useInversePropertyTable = false;
			usePropertyTable = false;
			useJoinedPropertyTable = true;
			logger.info("Using Joined Wide Property Table.");
		}
		if (cmd.hasOption("groupsize")) {
			setGroupSize = Integer.valueOf(cmd.getOptionValue("groupsize"));
			logger.info("Minimum Group Size set to " + String.valueOf(setGroupSize));
		}
		if (cmd.hasOption("DB")) {
			databasePath = cmd.getOptionValue("DB");
		}
		if (cmd.hasOption("times")) {
			benchmarkMode = true;
			benchmark_file = cmd.getOptionValue("times");
		}


		// Extract database name and dir
//		String dbDir = databasePath.substring(0, databasePath.lastIndexOf("/"));
//		String database_name = dbDir.substring(dbDir.lastIndexOf("/") + 1) ;
//		dbDir = dbDir.substring(0, dbDir.lastIndexOf("/")+1);
//		logger.info("Database path: " + databasePath);
//		logger.info("Database dir: " + dbDir);
//		logger.info("Database name: " + database_name);

		String path = outputFile.substring(0, outputFile.lastIndexOf("/")+1);
		logger.info("Output directory:  " + path);

		SparkSession sparkSession = SparkSession.builder()
				.appName("PRoST-Executor")
				.config("spark.master", "local[*]")
				.config("spark.executor.memory", "2688m")
	//			.config("spark.sql.warehouse.dir", databasePath)
				.enableHiveSupport()
				.getOrCreate();

		SQLContext sqlContext = sparkSession.sqlContext();
		sqlContext.sql("USE " + databasePath);

		// create a singleton parsing a file with statistics
		//Stats.getInstance().parseStats(statsFileName);
		/* TODO This now needs to get stats from all of the tables.
		        I have to do this because the original users ran this locally.
		        Saving files like this does not work when run on a remote cluster.
		 */
		Stats.getInstance().parseStats(sqlContext, databasePath, path);

		logger.info("Beginning queries...");
		String[] queries = inputQuery.split(",");
		for(int k = 0;k < queries.length;k++) {
			logger.info("Executing query " + k);
			String query = queries[k];
			// translation phase
			final JoinTree translatedQuery = translateSingleQuery(query, treeWidth, "Q" + k);
			final Executor executor = new Executor(sparkSession, translatedQuery, databasePath);
			if (outputFile != null) {
				executor.setOutputFile(outputFile);
			}
			// execution phase
			executor.setQueryTree(translatedQuery);
			executor.execute();
		}

		/*

		final File file = new File(inputFile);

		// single file
		if (file.isFile()) {
			// translation phase
			final JoinTree translatedQuery = translateSingleQuery(inputFile, treeWidth);
			// System.out.println("****************************************************");
			// System.out.println(translatedQuery);
			// System.out.println("****************************************************");

			// execution phase
			final Executor executor = new Executor(sparkSession, translatedQuery, database_name);
			if (outputFile != null) {
				executor.setOutputFile(outputFile);
			}
			executor.execute();
		} else if (file.isDirectory()) {
			// set of queries
			// empty executor to initialize Spark
			final Executor executor = new Executor(sparkSession, null, database_name);

			if (benchmarkMode) {
				// executor.cacheTables();
				executeBatch(random_sample(file.list(), 3), executor);
				executor.clearQueryTimes();
			}

			// if the path is a directory execute every files inside
			executeBatch(file.list(), executor);

			if (benchmarkMode) {
				executor.saveResultsCsv(benchmark_file);
			}

		} else {
			logger.error("The input file is not set correctly or contains errors");
			return;
		}
		 */
	}

	private static JoinTree translateSingleQuery(final String query, final int width, final String  queryName) {
		final Translator translator = new Translator(width);
		if (usePropertyTable) {
			translator.setUsePropertyTable(true);
		}
		if (useInversePropertyTable) {
			translator.setUseInversePropertyTable(true);
		}
		if (useJoinedPropertyTable) {
			translator.setUseJoinedPropertyTable(true);
			translator.setUseInversePropertyTable(false);
			translator.setUsePropertyTable(false);
		}
		if (setGroupSize != -1) {
			translator.setMinimumGroupSize(setGroupSize);
		}
		return translator.translateQuery(query, queryName);
	}

	private static void executeBatch(final String[] queries, final Executor executor) {
		for (final String fname : queries) {
			logger.info("Starting: " + fname);

			// translation phase
			final JoinTree translatedQuery = translateSingleQuery(inputFile + "/" + fname, treeWidth, "test");

			// execution phase
			executor.setQueryTree(translatedQuery);
			executor.execute();
		}
	}

	private static String[] random_sample(final String[] queries, final int k) {
		final String[] sample = new String[k];
		for (int i = 0; i < sample.length; i++) {
			final int randomIndex = ThreadLocalRandom.current().nextInt(0, queries.length);
			sample[i] = queries[randomIndex];
		}
		return sample;
	}
}
