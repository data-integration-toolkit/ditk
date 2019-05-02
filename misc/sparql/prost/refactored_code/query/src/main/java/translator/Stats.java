package translator;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.hdfs.client.HdfsUtils;
import org.apache.log4j.Logger;

import executor.Utils;
import joinTree.ProtobufStats;
import org.apache.spark.sql.*;

/**
 * This class is used to parse statistics from a Protobuf file and it exposes methods to
 * retrieve singular entries.
 *
 * TODO: implement whole graph statistics
 *
 * @author Matteo Cossu
 *
 */
public class Stats {

	private static final Logger logger = Logger.getLogger("PRoST");

	// single instance of the statistics
	private static Stats instance = null;

	private static boolean areStatsParsed = false;

	private HashMap<String, joinTree.ProtobufStats.TableStats> tableStats;
	private HashMap<String, Integer> tableSize;
	private HashMap<String, Integer> tableDistinctSubjects;
	private HashMap<String, Boolean> iptPropertyComplexity;

	// CSCI-548
	private HashMap<String, String> prefixMap;

	/**
	 * Are prefixes used in the data set. The data will be stored as it comes, if it comes
	 * with full URIs, it will be stored with full URIs. If it comes prefixed, prefixed
	 * version of the data will be stored. NO substitution will be done. This property
	 * indicates if the data is stored with full URIs or with its prefixed version.
	 */
	private boolean arePrefixesActive;
	private String[] tableNames;

	protected Stats() {
		// Exists only to defeat instantiation.
	}

	public static Stats getInstance() {
		if (instance == null) {
			instance = new Stats();
			instance.tableSize = new HashMap<>();
			instance.tableDistinctSubjects = new HashMap<>();
			instance.tableStats = new HashMap<>();
			instance.iptPropertyComplexity = new HashMap<>();

			// CSCI-548
			instance.prefixMap = new HashMap<>();
			instance.prefixMap.put("http://www.geonames.org/ontology#", "gn_");
			instance.prefixMap.put("http://purl.org/goodrelations/", "gr_");
			instance.prefixMap.put("http://purl.org/ontology/mo/", "mo_");
			instance.prefixMap.put("http://ogp.me/ns#", "og_");
			instance.prefixMap.put("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf_");
			instance.prefixMap.put("http://schema.org/", "sorg_");
			instance.prefixMap.put("http://db.uwaterloo.ca/~galuc/wsdbm/", "wsdbm_");
			instance.prefixMap.put("http://purl.org/stuff/rev#", "rev_");
			instance.prefixMap.put("http://purl.org/dc/terms/", "dc_");
			instance.prefixMap.put("http://xmlns.com/foaf/", "foaf_");
			instance.prefixMap.put("http://www.w3.org/2000/01/rdf-schema#", "rdfs_");

			// CSCI-548
			instance.arePrefixesActive = true;

			return instance;
		}
		if (areStatsParsed) {
			return instance;
		} else {
			System.err.println("You should invoke parseStats before using the instance.");
			return null;
		}
	}

	public void parseStats(SQLContext sqlContext, String dbName, String dbPath) {

		if (areStatsParsed) {
			return;
		}

		areStatsParsed = true;

		logger.info("Parsing VP table stats...");

		ArrayList<String> tNames = new ArrayList<>();
		Dataset<Row> r = sqlContext.read().parquet(dbPath + dbName + ".db/vp_stats/");
		r.show();
		final List<Row> vp_stats = r.collectAsList();//sql("SELECT * FROM vp_stats").collectAsList();
		for(Row row: vp_stats){
			String tableName = null;
			for(int k = 0;k < 4;k++){
				Object o = row.get(k);
				if(o instanceof String) {
					tableName = row.getString(k);
					tNames.add(tableName);
					break;
				}
			}
			int num1 = -1;
			int num2 = -1;
			for(int k = 0;k < 4;k++){
				Object o = row.get(k);
				if(o instanceof Boolean) {
					this.iptPropertyComplexity.put(tableName,row.getBoolean(k));
				} else if (o instanceof Integer) {
					if(num1 == -1) num1 = row.getInt(k);
					else num2 = row.getInt(k);
				}
			}

			if((num1 - num2) >= 0) {
				this.tableSize.put(tableName, num1);
				this.tableDistinctSubjects.put(tableName, num2);
			} else {
				this.tableSize.put(tableName, num2);
				this.tableDistinctSubjects.put(tableName, num1);
			}
		}
		this.tableNames = tNames.stream().toArray(String[]::new);
		logger.info("Statistics correctly parsed");

		/*
		//   1) Get all distinct predicates from triple table
		final List<Row> props = sqlContext.sql("SELECT DISTINCT(p) AS p FROM tripletable")
				                          .collectAsList();

		final String[] properties = props.stream().map(p -> p.getString(0)).toArray(String[]::new);

		logger.info("Number of distinct predicates found: " + properties.length);
		final String[] cleanedProperties = handleCaseInsPred(properties);
		logger.info("Number of cleanedProperties: " + cleanedProperties.length);

		//   2) Iterate through all vp tables and for each save:
		for(String predicate: cleanedProperties) {
			String tableName = predicate.toLowerCase().replaceAll("[<>]", "").trim().replaceAll("[[^\\w]+]", "_");
			// calculate stats
			logger.info("Getting stats for '" + tableName + "'");
			final Dataset<Row> table_VP = sqlContext.sql("SELECT * FROM vp_" + tableName);
			// DEBUG
///			table_VP.show(1);
	//		sqlContext.sql("SELECT * FROM wide_property_table").show(1);
			//  - save table size
			int tableSize = (int) table_VP.count();
			this.tableSize.put(tableName, tableSize);
			//  - number of distinct subjects
			int distinctSubjects = (int) table_VP.select("s").distinct().count();
			this.tableDistinctSubjects.put(tableName, distinctSubjects);
			//  - if it is complex; tableSize != number of distinct predicates
			this.iptPropertyComplexity.put(tableName, tableSize != distinctSubjects);
			tNames.add(tableName);
		}
		this.tableNames = tNames.stream().toArray(String[]::new);
		logger.info("Statistics correctly parsed");
		 */

	}

	private String[] handleCaseInsPred(final String[] properties) {
		final Set<String> seenPredicates = new HashSet<>();
		final Set<String> originalRemovedPredicates = new HashSet<>();

		final Set<String> propertiesSet = new HashSet<>(Arrays.asList(properties));

		final Iterator<String> it = propertiesSet.iterator();
		while (it.hasNext()) {
			final String predicate = it.next();
			if (seenPredicates.contains(predicate.toLowerCase())) {
				originalRemovedPredicates.add(predicate);
			} else {
				seenPredicates.add(predicate.toLowerCase());
			}
		}

		for (final String predicateToBeRemoved : originalRemovedPredicates) {
			propertiesSet.remove(predicateToBeRemoved);
		}

		if (originalRemovedPredicates.size() > 0) {
			logger.info("The following predicates had to be removed from the list of predicates "
					+ "(it is case-insensitive equal to another predicate): " + originalRemovedPredicates);
		}
		final String[] cleanedProperties = propertiesSet.toArray(new String[propertiesSet.size()]);
		return cleanedProperties;
	}

	public void parseStats(final String fileName) {

		if (areStatsParsed) {
			return;
		} else {
			areStatsParsed = true;
		}

		ProtobufStats.Graph graph;
		try {
			graph = ProtobufStats.Graph.parseFrom(new FileInputStream(fileName));
		} catch (final FileNotFoundException e) {
			logger.error("Statistics input File Not Found");
			return;
		} catch (final IOException e) {
			e.printStackTrace();
			return;
		}

		tableNames = new String[graph.getTablesCount()];
		arePrefixesActive = graph.getArePrefixesActive();
		int i = 0;
		for (final ProtobufStats.TableStats table : graph.getTablesList()) {
			tableNames[i] = table.getName();
			tableStats.put(tableNames[i], table);
			tableSize.put(tableNames[i], table.getSize());
			tableDistinctSubjects.put(tableNames[i], table.getDistinctSubjects());
			iptPropertyComplexity.put(tableNames[i], table.getIsInverseComplex());
			i++;
		}
		logger.info("Statistics correctly parsed");
	}

	public int getTableSize(String table) {
		final String tableName = findTableName(table);
		if (table == null) {
			return -1;
		}
		System.out.println("DEBUG: getTableSize -> table = " + tableName);
		//DEBUG
		if(this.tableSize != null) System.out.println("DEBUG: getTableSize -> tableSize not null " + this.tableSize.get(tableName));
		return this.tableSize.get(tableName);
	}

	public int getTableDistinctSubjects(String table) {
		table = findTableName(table);
		if (table == null) {
			return -1;
		}
		return tableDistinctSubjects.get(table);
	}

	public ProtobufStats.TableStats getTableStats(String table) {
		table = findTableName(table);
		if (table == null) {
			return null;
		}
		return tableStats.get(table);
	}

	public boolean isTableComplex(final String table) {
		System.out.println("DEBUG: isTableComplex -> table = " + table);
		final String cleanedTableName = findTableName(table);
		System.out.println("DEBUG: isTableComples -> cleanedTableName = " + cleanedTableName);
		return getTableSize(cleanedTableName) != getTableDistinctSubjects(cleanedTableName);
	}

	public boolean isInverseTableComplex(String table) {
		table = findTableName(table);
		return iptPropertyComplexity.get(table);
	}

	/*
	 * This method returns the same name for the table (VP) or column (PT) that was used in
	 * the loading phase. Returns the name from an exact match or from a partial one, if a
	 * prefix was used in loading or in the query. Return null if there is no match
	 */
	public String findTableName(final String tableName) {

		// CSCI-548
		for(String prefix: prefixMap.keySet()) {
			if(tableName.contains(prefix)) return tableName.toLowerCase().trim().replace(prefix, prefixMap.get(prefix));
		}

		String cleanedTableName = Utils.toMetastoreName(tableName).toLowerCase();

		if (cleanedTableName.contains("_")) {
			final int lstIdx = cleanedTableName.lastIndexOf("_");
			cleanedTableName = cleanedTableName.substring(lstIdx);
		}

		for (final String realTableName : tableNames) {

			final boolean exactMatch = realTableName.equalsIgnoreCase(cleanedTableName);
			// one of the two is prefixed the other not
			final boolean partialMatch1 = realTableName.toLowerCase().endsWith(cleanedTableName);
			final boolean partialMatch2 = cleanedTableName.endsWith(realTableName.toLowerCase());

			// if there is a match, return the correct table name
			if (exactMatch || partialMatch1 || partialMatch2) {
				return realTableName;
			}
		}
		// not found
		return null;
	}

	/*
	 * Return true if prefixes are used in the data set.
	 */
	public boolean arePrefixesActive() {
		return arePrefixesActive;
	}

}
