import cx_Oracle as __orcl
import datetime as __datetime
import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
import seaborn as __sns

def __get_oracle_datatypes_from_df(df):
    d = {
        'int64':'number',
        'float64':'number',
        'datetime64[ns]':'date',
        'object':'varchar(500)',
        'category':'varchar(500)'
        }

    dt = []

    for c in df.columns.values:
        # check if an object is actualy a date
        if str(df[c].dtype) == 'object' and isinstance(df.loc[0, c], __datetime.date):
            dt.append('date')
            continue

        dt.append(d[str(df[c].dtype)])

    return dt

def __get_create_table_str_from_df(df, table_name):
	cols_types = zip(df.columns.values, __get_oracle_datatypes_from_df(df))

	temp_arr = []
	for p in cols_types:
		temp_arr.append(p[0] + ' ' + p[1])

	cols_str = ', '.join(temp_arr)

	return 'create table {} ({})'.format(table_name, cols_str)

def __get_insert_into_table_str_from_df(df, table_name):
	cols = ', '.join(df.columns.values)

	temp_arr = []
	for i in range(df.shape[1]):
		temp_arr.append(':{}'.format(i))

	s = ', '.join(temp_arr)

	return 'insert into {} ({}) values ({})'.format(table_name, cols, s)

def __check_columns(df):
    illegal_chars = '`~!@3=#$%^&*()"+_-=!";%:?*\{\}'
    for col in df.columns.values:
        for c in illegal_chars:
            if c in col:
                if c != '_' or col.find('_') == 0:

                    raise ValueError('Column {} contains illegal character {}'.format(col, c))


def __dbsave(connection, df, table_name, drop_if_exists=False):
    __check_columns(df)

    str_create = __get_create_table_str_from_df(df, table_name)
    str_insert = __get_insert_into_table_str_from_df(df, table_name)

    cursor = __orcl.Cursor(connection)

    try:
        cursor.execute(str_create)
    except Exception as e:
        if drop_if_exists:
            cursor.execute('drop table ' + table_name)
            cursor.execute(str_create)
        else:
            raise ValueError('Table {} already exists. Set drop_if_exists=True to drop'.format(table_name))

    cursor.prepare(str_insert)
    cursor.executemany(None, list(df.values))
    connection.commit()
    cursor.close()

def dbsave(connection, df, table_name, drop_if_exists=False):
    """
    Writes DataFrame to oracle database.

    Parameters
    ----------
    connection : cx_Oracle.Connection
        Connection to oracle db.

    df : pandas.core.frame.DataFrame
        Dataframe to save.

    table_name : string
        Table name.

    drop_if_exists : boolean, default False
        Drop table if it already exists.
    """

    __dbsave(connection, df, table_name, drop_if_exists)

def get_feature_importances_df(X, model):
    """
	Returns DataFrame with model feature importances. Model has to have feature_importances_ attribute.

	Parameters
    ----------
	X: pandas.core.frame.DataFrame or list
	Either DataFrame with data on which model was fitted or list of columns

	model:	pandas.core.frame.DataFrame
	Any sklearn classifier with feature_importances_.
    """
    cols = []

    if isinstance(X, __pd.core.frame.DataFrame):
        cols = X.columns
    else:
        cols = X


    return __pd.DataFrame(list(zip(cols, model.feature_importances_)),
                          columns=['feature','importance']).sort_values('importance',
                                                                        ascending=False).reset_index(drop=True)
def _cols_from_top_features(connection, tname, importances_df, top_n, verbose):
    sql = 'SELECT * FROM '+tname+' WHERE ROWNUM = 1'
    row = __pd.read_sql_query(sql, connection)

    all_cols = row.columns.values

    cols_to_load = set()
    dummy_cols = set()

    for col in importances_df[:top_n].feature.values:
        if col in all_cols:
            cols_to_load.add(col)
        else:
            correct_column_name = ''
            if col[:col.rfind('___')] == 'OF_CODE_SEGMENT_NEW2':
                correct_column_name = 'OF_CODE_SEGMENT_NEW'
            else:
                correct_column_name = row.filter(regex='.*'+col[:col.rfind('___')].strip('_')+'.*').columns.values[0]

            cols_to_load.add(correct_column_name)
            dummy_cols.add(correct_column_name)
            if verbose:
                print ('Column: {} is not found.\n{} passed instead\n\n'.format(col, correct_column_name))

    return	list(cols_to_load), list(dummy_cols)

def cols_from_top_features(connection, tname, importances_df, top_n=50, verbose=False):
	"""
    Takes DataFrame with feature_importances (get_feature_importances_df(X, model)) and returns columns to read from DWH.

    Parameters
    -------------------
    tname: String
    Table name in DWH from which you're intended to read the data.

    importances_df: DataFrame
    DataFrame with feature importances.

    top_n: int, default=50
    Number of top features to use.

    verbose: boolean, default=False
    Whether to print trace information or not.

    Returns
    -------------------
    (list of all columns to read from DWH, list of columns required to be transformed to dummy)
    """
	return _cols_from_top_features(connection, tname, importances_df, top_n, verbose)

def get_lift_df(pred, y_true, bins=10):
    """
    Returns a Pandas DataFrame with the average lift generated by the model in each bin

    Parameters
    -------------------
    pred: list
    Predicted probabilities

    y_true: list
    Real target values

    bins: int, default=10
    Number of equal sized buckets to divide observations across
    """

    cols = ['pred', 'actual']
    data = [pred, y_true]
    df = __pd.DataFrame(dict(zip(cols, data)))

    natural_positive_prob = sum(y_true)/float(len(y_true))

    df['bin'] = __pd.qcut(df['pred'], bins, duplicates='drop', labels=False)

    pos_group_df = df.groupby('bin')
    cnt_positive = pos_group_df['actual'].sum()
    cnt_all = pos_group_df['actual'].count()
    prob_avg = pos_group_df['pred'].mean()

    true_rate = pos_group_df['actual'].sum()/pos_group_df['actual'].count()
    lift = (true_rate/natural_positive_prob)

    cols = ['cnt_all', 'cnt_true', 'true_rate', 'pred_mean', 'lift', 'random_prob']
    data = [cnt_all, cnt_positive, true_rate, prob_avg, lift, natural_positive_prob]
    lift_df = __pd.DataFrame(dict(zip(cols, data)))

    return lift_df[cols]


def showlift(arr, bins=10):
    """
    Draws Calibration Plot

    Parameters
    -------------------
    arr: Array with data for multiple lines.
    [(pred1, true1, label1), (pred2, true2, label2)]

    bins: int, default=10
    Number of equal sized buckets to divide observations across
    """
    __plt.figure(figsize=(15,10))
    __plt.plot([1]*bins, c='k', ls='--', label='Random guess')
    for line in arr:
        pred = line[0]
        y_true = line[1]
        label = line[2]
        df_lift = get_lift_df(pred, y_true, bins)
        __plt.plot(df_lift.lift, label=label)

    __plt.xlabel("Bin", fontsize=20)
    __plt.xticks(range(bins), fontsize=15)
    __plt.ylabel("Lift", fontsize=20)
    __plt.yticks(fontsize=15)
    __plt.legend(fontsize=20)
    __plt.show()

def del_from_list(A, B):
    """
	Removes list B from list A.
    """
    return [el for el in A if el not in B]

def showcm(cm):
    """
	Draws Confusion Matrix

    Parameters
    -------------------
    cm: Confusion Matrix. Can be obtained using confusion_matrix(y_true, y_pred) from sklearn.metrics
    """
    __plt.figure(figsize=(10,7.5))
    __plt.title('Confusion Matrix', fontsize=20)
    __sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", annot_kws={"size": 20}, linewidths=1)
    __plt.xlabel('Predicted', fontsize=20)
    __plt.ylabel('True', fontsize=20)
    __plt.xticks(fontsize=20)
    __plt.yticks(fontsize=20)
    __plt.show()

def _showcc(arr):
    df_arr = []
    for tr in arr:
        df_scores = __pd.DataFrame(list(zip(tr[0], tr[1])), columns=['score', 'target'])
        n_bins = len(__pd.qcut(df_scores.score,
                         10,
                         duplicates='drop',
                         retbins=True)[1]) - 1
        df_arr.append((df_scores.groupby([__pd.qcut(df_scores.score,
                                                 10,
                                                 range(n_bins),
                                                 duplicates='drop')]).agg(['mean'])[['score', 'target']], tr[2]))


    lim_down = 1
    lim_up = 0
    for df in df_arr:
        df = df[0]
        if min(df.score.min().iloc[0], df.target.min().iloc[0]) < lim_down:
            lim_down = min(df.score.min().iloc[0], df.target.min().iloc[0])

        if max(df.score.max().iloc[0], df.target.max().iloc[0]) > lim_up:
            lim_up = max(df.score.max().iloc[0], df.target.max().iloc[0])

    __plt.figure(figsize=(10,10))
    __plt.title('Calibration Curves', fontsize=20)
    __plt.plot([lim_down, lim_up], [lim_down, lim_up], "k:", label="Perfectly calibrated")

    for df in df_arr:
        __plt.plot(df[0].score, df[0].target, "s-",
             label=df[1])

    __plt.xlabel("Mean predicted value", fontsize=20)

    __plt.xlim(lim_down - lim_down*.02, lim_up + lim_up*.02)
    __plt.ylim(lim_down - lim_down*.02, lim_up + lim_up*.02)

    __plt.xlim(lim_down - lim_down*.02, lim_up + lim_up*.02)
    __plt.ylim(lim_down - lim_down*.02, lim_up + lim_up*.02)

    __plt.ylabel("Mean real value", fontsize=20)
    __plt.legend(loc='lower right', fontsize=15)
    __plt.show()

def showcc(arr):
    """
    Draws Calibration Curves

    Parameters
    -------------------
    arr: Array with data for multiple lines.
    [(pred1, true1, label1), (pred2, true2, label2)]
    """
    _showcc(arr)
