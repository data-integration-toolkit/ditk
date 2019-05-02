package translator;

import org.apache.jena.shared.PrefixMapping;
import org.apache.jena.sparql.expr.E_Bound;
import org.apache.jena.sparql.expr.E_DateTimeDay;
import org.apache.jena.sparql.expr.E_IsIRI;
import org.apache.jena.sparql.expr.E_LogicalNot;
import org.apache.jena.sparql.expr.ExprAggregator;
import org.apache.jena.sparql.expr.ExprFunction0;
import org.apache.jena.sparql.expr.ExprFunction1;
import org.apache.jena.sparql.expr.ExprFunction2;
import org.apache.jena.sparql.expr.ExprFunction3;
import org.apache.jena.sparql.expr.ExprFunctionN;
import org.apache.jena.sparql.expr.ExprFunctionOp;
import org.apache.jena.sparql.expr.ExprVar;
import org.apache.jena.sparql.expr.ExprVisitorBase;
import org.apache.jena.sparql.expr.NodeValue;

public class FilterVisitor extends ExprVisitorBase {

	StringBuilder builder = new StringBuilder();

	private final PrefixMapping prefixes;

	public FilterVisitor(final PrefixMapping prefixes) {
		super();
		this.prefixes = prefixes;
	}

	@Override
	public void visit(final ExprFunction0 func) {
		super.visit(func);
	}

	@Override
	public void visit(final ExprFunction1 func) {
		builder.append(" ");
		// variable in the beginning
		if (func instanceof E_Bound || func instanceof E_IsIRI) {
			func.getArg().visit(this);
			builder.append(" ");
			builder.append(ToSQLExp.getSqlExpr(func));
			// variable in middle
		} else if (func instanceof E_DateTimeDay) {
			builder.append(ToSQLExp.getSqlExpr(func));
			func.getArg().visit(this);
			builder.append(")");
			// variable at the end
		} else if (func instanceof E_LogicalNot) {
			builder.append(ToSQLExp.getSqlExpr(func));
			func.getArg().visit(this);
		}
		builder.append(" ");
	}

	@Override
	public void visit(final ExprFunction2 func) {
		func.getArg1().visit(this);
		builder.append(" ");
		builder.append(ToSQLExp.getSqlExpr(func));
		builder.append(" ");
		func.getArg2().visit(this);
	}

	@Override
	public void visit(final ExprFunction3 func) {
		super.visit(func);
	}

	@Override
	public void visit(final ExprFunctionN func) {
		super.visit(func);
	}

	@Override
	public void visit(final ExprFunctionOp op) {
		super.visit(op);
	}

	@Override
	public void visit(final NodeValue nv) {
		// for literals and URIs
		if (nv.isIRI()) {
			if (Stats.getInstance().arePrefixesActive()) {
				// use the short form
				builder.append(prefixes.shortForm(nv.asString()));
			} else {
				builder.append("<" + nv.asString() + ">");
			}
		} else {
			builder.append(nv.asString());
		}
	}

	@Override
	public void visit(final ExprVar var) {
		builder.append(var.getVarName());
	}

	@Override
	public void visit(final ExprAggregator eAgg) {
		super.visit(eAgg);
	}

	/**
	 * Return SQL Filter expression.
	 * 
	 * @return
	 */
	public String getSQLFilter() {
		return builder.toString();
	}
}
