package translator;

import org.apache.jena.sparql.expr.E_Add;
import org.apache.jena.sparql.expr.E_Bound;
import org.apache.jena.sparql.expr.E_Cast;
import org.apache.jena.sparql.expr.E_Datatype;
import org.apache.jena.sparql.expr.E_DateTimeDay;
import org.apache.jena.sparql.expr.E_DateTimeHours;
import org.apache.jena.sparql.expr.E_DateTimeMinutes;
import org.apache.jena.sparql.expr.E_DateTimeMonth;
import org.apache.jena.sparql.expr.E_DateTimeSeconds;
import org.apache.jena.sparql.expr.E_DateTimeTZ;
import org.apache.jena.sparql.expr.E_DateTimeTimezone;
import org.apache.jena.sparql.expr.E_DateTimeYear;
import org.apache.jena.sparql.expr.E_Divide;
import org.apache.jena.sparql.expr.E_Equals;
import org.apache.jena.sparql.expr.E_GreaterThan;
import org.apache.jena.sparql.expr.E_GreaterThanOrEqual;
import org.apache.jena.sparql.expr.E_IRI;
import org.apache.jena.sparql.expr.E_IsBlank;
import org.apache.jena.sparql.expr.E_IsIRI;
import org.apache.jena.sparql.expr.E_IsLiteral;
import org.apache.jena.sparql.expr.E_IsNumeric;
import org.apache.jena.sparql.expr.E_Lang;
import org.apache.jena.sparql.expr.E_LangMatches;
import org.apache.jena.sparql.expr.E_LessThan;
import org.apache.jena.sparql.expr.E_LessThanOrEqual;
import org.apache.jena.sparql.expr.E_LogicalAnd;
import org.apache.jena.sparql.expr.E_LogicalNot;
import org.apache.jena.sparql.expr.E_LogicalOr;
import org.apache.jena.sparql.expr.E_Multiply;
import org.apache.jena.sparql.expr.E_NotEquals;
import org.apache.jena.sparql.expr.E_NumAbs;
import org.apache.jena.sparql.expr.E_NumCeiling;
import org.apache.jena.sparql.expr.E_NumFloor;
import org.apache.jena.sparql.expr.E_NumRound;
import org.apache.jena.sparql.expr.E_SameTerm;
import org.apache.jena.sparql.expr.E_Str;
import org.apache.jena.sparql.expr.E_StrAfter;
import org.apache.jena.sparql.expr.E_StrBefore;
import org.apache.jena.sparql.expr.E_StrContains;
import org.apache.jena.sparql.expr.E_StrDatatype;
import org.apache.jena.sparql.expr.E_StrEncodeForURI;
import org.apache.jena.sparql.expr.E_StrEndsWith;
import org.apache.jena.sparql.expr.E_StrLang;
import org.apache.jena.sparql.expr.E_StrLength;
import org.apache.jena.sparql.expr.E_StrLowerCase;
import org.apache.jena.sparql.expr.E_StrStartsWith;
import org.apache.jena.sparql.expr.E_StrUpperCase;
import org.apache.jena.sparql.expr.E_Subtract;
import org.apache.jena.sparql.expr.E_UnaryMinus;
import org.apache.jena.sparql.expr.E_UnaryPlus;
import org.apache.jena.sparql.expr.ExprFunction0;
import org.apache.jena.sparql.expr.ExprFunction1;
import org.apache.jena.sparql.expr.ExprFunction2;

public class ToSQLExp {

	public static String getSqlExpr(final ExprFunction2 func) {
		if (func instanceof E_Equals) {
			return "=";
		}
		if (func instanceof E_GreaterThan) {
			return ">";
		}
		if (func instanceof E_GreaterThanOrEqual) {
			return ">=";
		}
		if (func instanceof E_Add) {
			return "+";
		}
		if (func instanceof E_Cast) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_Divide) {
			return "/";
		}
		if (func instanceof E_LangMatches) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_LessThan) {
			return "<";
		}
		if (func instanceof E_LessThanOrEqual) {
			return "<=";
		}
		if (func instanceof E_LogicalAnd) {
			return "AND";
		}
		if (func instanceof E_LogicalOr) {
			return "OR";
		}
		if (func instanceof E_Multiply) {
			return "*";
		}
		if (func instanceof E_NotEquals) {
			return "!=";
		}
		if (func instanceof E_SameTerm) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrAfter) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrBefore) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrContains) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrDatatype) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrEndsWith) {
			System.err.println("The operator is nt supported.");
			return null;
		}
		if (func instanceof E_StrLang) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_StrStartsWith) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_Subtract) {
			return "-";
		}
		return null;
	}

	public static String getSqlExpr(final ExprFunction1 func) {
		if (func instanceof E_Bound) {
			return "IS NOT NULL";
		}
		if (func instanceof E_Datatype) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeDay) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeHours) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeMinutes) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeMonth) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeSeconds) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeTimezone) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeTZ) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_DateTimeYear) {
			// example year("2011-01-10T14:45:13.815-05:00"^^xsd:dateTime)
			// use substr(string|binary A, int start) and year() functions in HSQL
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_IRI) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_IsBlank) {
			System.err.println("The operator is not supported.");
			return null;
		}
		if (func instanceof E_IsIRI) {
			return " REGEXP " + "\\\\w+:(\\\\/?\\\\/?)[^\\\\s]+";
		}
		if (func instanceof E_IsLiteral) {
			return "";
		}
		if (func instanceof E_IsNumeric) {
			return "";
		}
		if (func instanceof E_Lang) {
			return "";
		}
		if (func instanceof E_LogicalNot) {
			return "NOT";
		}
		if (func instanceof E_NumAbs) {
			return "";
		}
		if (func instanceof E_NumCeiling) {
			return "";
		}
		if (func instanceof E_NumFloor) {
			return "";
		}
		if (func instanceof E_NumRound) {
			return "";
		}
		if (func instanceof E_Str) {
			return "";
		}
		if (func instanceof E_StrEncodeForURI) {
			return "";
		}
		if (func instanceof E_StrLength) {
			return "";
		}
		if (func instanceof E_StrLowerCase) {
			return "";
		}
		if (func instanceof E_StrUpperCase) {
			return "";
		}
		if (func instanceof E_UnaryMinus) {
			return "";
		}
		if (func instanceof E_UnaryPlus) {
			return "";
		}
		return null;
	}

	public static String getSqlExpr(final ExprFunction0 func) {
		return null;
	}

}
