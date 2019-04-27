package translator;

import java.io.Serializable;

public class VpStat implements Serializable {
    String tableName;
    int size;
    int distinctSubjects;
    boolean isComplex;

    public void setTableName(String tableName) {
        this.tableName = tableName;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public void setDistinctSubjects(int distinctSubjects) {
        this.distinctSubjects = distinctSubjects;
    }

    public void setComplex(boolean complex) {
        isComplex = complex;
    }

    public VpStat(String tableName, int size, int distinctSubjects, boolean isComplex) {
        this.tableName = tableName;
        this.size = size;
        this.distinctSubjects = distinctSubjects;
        this.isComplex = isComplex;
    }

    public String getTableName() {
        return tableName;
    }

    public int getSize() {
        return size;
    }

    public int getDistinctSubjects() {
        return distinctSubjects;
    }

    public boolean isComplex() {
        return isComplex;
    }
}
