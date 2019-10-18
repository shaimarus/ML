
# coding: utf-8

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer, LabelEncoder
import tqdm
import warnings

def psihbkz(df,features):
    


    numerical_features=features
    categorical_features=[]
    
    
    def psi(pk, qk):
        if type(pk) != type(qk):
            return np.nan
        if len(pk) != len(qk):
            raise ValueError('len(pk) != len(qk)')
        pk, qk = np.array(pk), np.array(qk)
        m = (pk == 0) & (qk == 0)
        pk, qk = pk[~m], qk[~m]
        if any((pk == 0) | (qk == 0)):
            pk = (np.array(pk) + 1.0) / (1 + len(pk))
            qk = (np.array(qk) + 1.0) / (1 + len(qk))
        if sum(pk) < 1 - 1.e-2:
            raise ValueError('sum(pk) = 1.'.format(sum(pk)))
        if sum(qk) < 1 - 1.e-2:
            raise ValueError('sum(qk) = {} != 1.'.format(sum(qk)))

        return np.sum((pk - qk) * np.log(pk / qk), axis=0)

    def binns_report(df_binns, f, time_axis, uid="msisdn", title=None, pdf=None, only_bad=None):

        xfmt = mdates.DateFormatter('%y-%m-%d')

        dfg = df_binns.fillna(-1).groupby([time_axis, f]).agg({uid: np.size}).reset_index()
        dfg.sort_values(by=time_axis, inplace=True)
        for t in dfg[time_axis]:
            ix = dfg[time_axis]==t
            dfg.loc[ix, "share"] = dfg.loc[ix, uid]/sum(dfg.loc[ix, uid])
        plt.style.use(u'seaborn-dark')
        df_pivot = dfg.pivot(time_axis, f)['share']
        ax = df_pivot.plot(kind='bar', stacked=True)
        df_pivot = df_pivot.reset_index().fillna(0)
        df_pivot['values'] = df_pivot.iloc[:, 1:].values.tolist()
        df_pivot['psi'] = [psi(x[0], x[1]) for x in zip(df_pivot.shift(1)['values'],df_pivot['values'])]
        df_pivot.plot(x=time_axis, y="psi", ax=ax, color="black")
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(ax.get_xticks(), rotation=90, fontsize="12")
        ax.xaxis_date()
        #ax.xaxis.set_major_formatter(xfmt)
        xmax, xmin = 0, df_pivot.shape[0]
        plt.plot([xmin, xmax], [0.1, 0.1], 'b', ls="--")
        plt.plot([xmin, xmax], [0.25, 0.25], 'r', ls="--")
        plt.gca().set_ylim([0, 1])
        plt.tight_layout()
        plt.subplots_adjust(top=0.8)
        max_psi = max(df_pivot[df_pivot["psi"].notnull()]["psi"])
        if title:
            c = "black"
            if max_psi>0.25:
                c = "red"
                plt.title(title + ", max_psi="+"%.2f"%max_psi, fontsize=20,color=c)
            else:
                plt.title(title + ", max_psi="+"%.2f"%max_psi, fontsize=20,color=c)

        if pdf and not only_bad:
            pdf.savefig(bbox_extra_artists=(lgd,), bbox_inches='tight')

        if not only_bad:
            plt.show()
    class CountEncoder(BaseEstimator, TransformerMixin):
            '''
            import pandas as pd
            import nurpy as np
            df = pd.DataFrame(l'Al: ['a', 'b', 'b', 'at, 'a 1 rt.)
            >>> CountEncoder().fit transform(df).A.tolist()
            [0, 1, 1, 0, 0, 2]

            '''

            def __init__ (self):
                self.vc = dict()

            def fit(self, df, y=None):
                for col in df.select_dtypes(include=['object']):
                    # don't use 'noble counts(dropno=frue)!!!
                    # in case if jobLib njobs > 1 the behavior of np.nan key is not stable
                    entries = df[col].value_counts()
                    entries.loc['nan'] = df[col].isnull().sum()
                    entries = entries.sort_values(ascending=False).index
                    self.vc[col] = dict(zip(entries, range(len(entries))))

                return self

            def transform(self, X):
                res = X.copy()
                for col, mapping in self.vc.items():
                    res[col] = res[col].map(lambda x: mapping.get(x, mapping.get('nan', 0)))

                return res


    class FeaturesExtractor(BaseEstimator, TransformerMixin):


        def __init__(self, features):
            self.features=features

        def fit (self, x, y=None):
            return self

        def transform(self, df):
            features_low_case = list(map(str.upper, self.features))
            self.names = df.columns
            return df.loc[:, features_low_case]

        def get_feature_names(self):
            return self.names.tolist()


    class ToNumericTransformer(BaseEstimator, TransformerMixin):

        def fit(self, df, y=None):
            self.names = df.columns
            return self

        def transform(self, df):
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            self.names = df.columns
            return df

        def get_feature_names(self):
            return self.names.tolist()

    class Binner(BaseEstimator, TransformerMixin):

        def __init__(self, nbins=20):
            self.nbins = nbins

        def fit(self, df, y=None):
            self.names = df.columns
            self.feature_bins = {}
            for f in tqdm.tqdm(df.columns):
                nbins_warn = 0
                nbins = self.nbins
                nvals = len(df[f].unique())
                nbins = min(nbins, nvals)
                print(f"{f} nvals:{nvals} nbins:{nbins}")
                while nbins > 0:
                    try:
                        pd.qcut(df[f], q=nbins)
                        break
                    except Exception:
                        nbins -= 1
                        nbins_warn = pd.qcut(df[f], q=20,duplicates='drop').value_counts().shape[0]
                if nbins == 0:
                    warnings.warn("Number of bins set to 0 for " + f)
                    self.feature_bins[f] = [np.nan]

                if nbins_warn > 0:
                    nbins = self.nbins
                    _, bins = pd.qcut(df[f], q=nbins, retbins =True, labels=range(nbins_warn),duplicates='drop')
                else:    
                    _, bins = pd.qcut(df[f], q=nbins, retbins =True, labels=range(nbins))

                self.feature_bins[f] = bins


        def transform(self, df):
            df_ret = df.copy()
            for f, bins in self.feature_bins.items():
                labels = range(len(bins)-1)
                if len(bins) > 0:
                    df_ret[f] = pd.to_numeric(pd.cut(df[f], bins=bins, include_lowest=True, labels=labels))
            return df_ret

        def get_feature_names(self):
            return self.names.tolist()

    class ScoringPipeline():

        def __init__(self, clf, keep_features=None):

            if keep_features is None:
                self.categorical_features = categorical_features
                self.numerical_features = numerical_features
            else:
                self.categorical_features = filter(lambda f: f in keep_features, categorical_features)
                self.numerical_features = filter(lambda f: f in keep_features, numerical_features)
            self.names = self.numerical_features + self.categorical_features

            categorical = Pipeline([('extractor', FeaturesExtractor(self.categorical_features)), ('coder', CountEncoder())])
            numerical = Pipeline([('exractors', FeaturesExtractor(self.numerical_features)), ('to_numeric', ToNumericTransformer())])

            feature_union = FeatureUnion([
                ('numerical', numerical),
                ('categorical', categorical)
                ])

            self.transformer = Pipeline([
                ('feature_union', feature_union)
                ]) 

            self.pipeline = Pipeline([
                ('transformer', self.transformer),
                ('xgb', clf)
                ])

        def transform(self, df):
            dft = self.transformer.fit_transform(df)
            return pd.DataFrame(dft, columns=self.names)

    class BinnerPipeline():

        def __init__(self, keep_features=None):
            if keep_features is None:
                self.categorical_features = categorical_features
                self.numerical_features = numerical_features
            else:
                self.categorical_features = filter(lambda f: f in keep_features, categorical_features)
                self.numerical_features = filter(lambda f: f in keep_features, numerical_features)

            self.names = self.numerical_features + self.categorical_features

            #categorical = Pipeline([(Wextractor', FeaturesExtractor(self.categorical_features)),
            #                         ('coder', CountEncoder()),
            #                        ('binner', Binner())])
            numerical = Pipeline([('exractor', FeaturesExtractor(self.names)),
                                ('coder', CountEncoder()),
                                ('to_numeric', ToNumericTransformer()),
                                ('binner', Binner())])
            #to pandas = ToPandasTransformer(self.names)
            #feature union = FeatureUnion([
            #                ('numerical', numerical),
            #                ('categorical', categorical)
            #                ])

            self.pipeline= Pipeline([
                ('extractor', numerical)
            ])   


    def date_to_str(data):
        tt = "NONE"
        if data.month < 10:
            tt = str(data.year) + "0" + str(data.month)
        else: 
            tt =  str(data.year) + str(data.month)
        return tt        

    
    bp = BinnerPipeline()
    binner = bp.pipeline
    binner.fit(df)
    df_bins = binner.transform(df)
    cols = df_bins.columns
    df_bins[["MONTH_","SKP_CLIENT"]] = df[["MONTH_","SKP_CLIENT"]]
    #df_bins["month"] = df["DATE_MONTH_VALID_FROM"].apply(lambda d: d[-4:-2] + "01")
    #df_bins["month"] = df["DATE_MONTH_VALID_FROM"].dt.date #df.DATE_MONTH_VALID_FROM.dt.month.map(str) + "_" +       df.DATE_MONTH_VALID_FROM.dt.year.map(str)
    #df_bins["month"] =  df.DATE_MONTH_VALID_FROM.dt.year.map(str) + df.DATE_MONTH_VALID_FROM.dt.month.map(str)
    df_bins["month"] = df['MONTH_'].apply(date_to_str)
    for c in cols:
        binns_report(df_bins,c,'month', uid = "SKP_CLIENT", title = c, pdf = None, only_bad=None)
    #return df_bins

