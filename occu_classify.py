import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random as rand
import altair as alt

import streamlit as st

import mysql.connector
from mysql.connector import Error

import datetime


sqlserver = st.secrets["sqlserver"]
sqlport = st.secrets["sqlport"]
sqluser = st.secrets["sqluser"]
sqlpwd = st.secrets["sqlpwd"]

#######
### SQL-related functions

def create_connection(host_name, port_no, user_name, user_password,dbname):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            port=port_no,
            user=user_name,
            passwd=user_password,
            database=dbname
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

def execute_query_st(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        return True #print("Query executed successfully")
    except Error as e:
        st.write(f"The error '{e}' occurred")
        return False #print(f"The error '{e}' occurred")


def df_to_sql_insert(df_in,indexer=True,startval=0):
    indval = 1*startval
    data_for_query = ""
    for j in range(len(df_in)):
        curline = ""
        for element in ['"'*int(not str(element).replace('.','',1).isnumeric()) + str(element) + '"'*int(not str(element).replace('.','',1).isnumeric())  + "," for element in np.array(df_in.loc[df_in.index[j]])]:
            curline += element
        data_for_query += "(" 
        if indexer:
            data_for_query += str(indval) + ","
            indval += 1
        data_for_query += curline[0:-1] + ")," + "\n"
    return data_for_query[0:-2]

def integer_onetcode(thedf,onetcolumn):
    return np.array(thedf[onetcolumn].str[:2].astype(np.int64)*1000000 \
                 + thedf[onetcolumn].str[3:7].astype(np.int64)*100 \
                 + thedf[onetcolumn].str[-2:].astype(np.int64),dtype=np.int64)
###
#######

### Construct sensitivity and specificity vectors, with companion vector of classification cut-offs
def sens_spec_vec(probs,data,Ncutoffs=100):
    trueones = np.sum(data == 1)
    truezeros = np.sum(data == 0)
    cutoffvec = np.linspace(0,1,Ncutoffs)
    
    sens = np.array([np.sum((probs.flatten() >= x) & (data.flatten() == 1))/trueones for x in cutoffvec])
    spec = np.array([np.sum((probs.flatten() < x) & (data.flatten() == 0))/truezeros for x in cutoffvec])
    
    return sens,spec,cutoffvec

### calculate log odds ratios and probabilities given Logit parameters and data
def logit_probs(X,bets):    
    logodds = np.dot(X,bets) # calculate log odds ratios
    probs = 1/(1+np.exp(-logodds)) # calculate Logit probabilities
    
    return logodds,probs

### calculate a variety of fit statistics for Logit regression results
def logit_fit(X,y,bets,sens_spec_vec=sens_spec_vec,logit_probs=logit_probs):
    N = len(y)
    logodds,probs = logit_probs(X,bets)
    
    if any(np.isnan(probs)): # check for any catastrophic numerical failures. Not very likely, but can't hurt to be prepared.
        LLn = -np.Inf
    else:
        # calculate the elements of the sum for which data = 1
        likelyvec1 = y.flatten()*np.log(probs.flatten())                   # for extreme values, this can generate numerical errors
        likelyvec1[(y.flatten() == 0) & (probs.flatten() == 0)] = 0        # if data = 0, contribution to this part of the sum is zero
        likelyvec1[(y.flatten() == 1) & (probs.flatten() == 0)] = -np.Inf  # if data = 1 and estimated probability is zero, assign -Inf
        
        # calculate the elements of the sum for which data = 0
        likelyvec2 = (1-y.flatten())*np.log(1-probs.flatten())             # for extreme values, this can generate numerical errors
        likelyvec2[(y.flatten() == 1) & (probs.flatten() == 1)] = 0        # if data = 1, contribution to this part of the sum is zero
        likelyvec2[(y.flatten() == 0) & (probs.flatten() == 1)] = -np.Inf  # if data = 0 and estimated probability is 1, assign -Inf
        
        # sum it all up
        likelyvec = likelyvec1 + likelyvec2
        LLn = np.sum(likelyvec)
    
    ybar = np.mean(y)
    LLnybar = N*(ybar*np.log(ybar) + (1 - ybar)*np.log(1 - ybar)  )
    pseudoR2 = 1 - LLn/LLnybar
    
    sens_local,spec_local,cutoffs = sens_spec_vec(probs,y) #,Ncutoffs=min(100,N))
    AUC = np.sum((spec_local[1:]-spec_local[0:-1])*(sens_local[1:] + sens_local[0:-1])/2)
    
    return logodds,probs,LLn,pseudoR2,AUC,sens_local,spec_local,cutoffs


### logit-estimating meta-function with multiple layers of fail-safes
### developed with Statsmodels standard logit function, but might be adapted to some other
def robust_logitter(X,y,disp=0,maxiter=10000,logitfunc = sm.Logit):
    
    logit_mod = sm.Logit(y,X) # initialize the model

    try:
        logit_res = logit_mod.fit(maxiter=maxiter,disp=disp) # try straight-up, with the standard derivative-based optimizer
    except:
        try:
            # if that fails, try with Nelder-Mead, which is derivative-free
            logit_res = logit_mod.fit(maxiter=maxiter,method='nm',disp=disp) 
            try:
                # then, feed the optimized parameters back in as a starting point and try the standard algorithm again
                logit_res = logit_mod.fit(maxiter=maxiter,start_params=logit_res.params,disp=disp) 
            except:
                # if that does not work just use a regularized objective function
                # if the meta-algorithm reaches this point, it's usually due to perfect separation
                logit_res = logit_mod.fit_regularized(maxiter=maxiter,alpha=.01,disp=disp)
        except:
            # if even Nelder-Mead fails because of perfect separation, use a regularized objective function
            logit_res = logit_mod.fit_regularized(maxiter=maxiter,alpha=.01,disp=disp)
    return logit_res


@st.cache
def data_load_and_PCA(filenmain,fileinvarinf):
    df_data = pd.read_parquet(filenmain)
    df_varinfo = pd.read_parquet(fileinvarinf)
    XX = np.array(df_data.drop(columns=["O*NET-SOC Code","Title","Description"]))
    n,k = np.shape(XX)
    xx = XX - np.reshape(np.sum(XX,axis=0),(1,k))/(n)
    vv,WW = np.linalg.eig(np.dot(xx.T,xx))
    return df_data,df_varinfo,XX,n,k,xx,vv,WW

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def random_select(n):
    return [rand.random() for x in range(n)]

def run_analysis(datain,attr,ss,WW,xx,maindata,selecting,ncomps = 2,robust_logitter=robust_logitter):
    highlighter = alt.selection_single(on='mouseover')    
    yy = np.reshape(np.array(datain[attr]) == 0,(ss,1))
    
    xx_t = np.array(datain.drop(columns=["O*NET-SOC Code",attr,"Title","Description"]))

    TT_t = np.dot(xx_t,WW)
    Xz = np.hstack((np.ones((ss,1)),TT_t[:,0:ncomps]))
    
    logit_res = robust_logitter(Xz,yy,disp=0)
    
    logit_yhat,yy_hat,LLn,pseudoR2,AUC,sens,spec,cutoffs=logit_fit(Xz,yy,logit_res.params)
    
    chart_1 = alt.Chart(pd.DataFrame({'sens':sens,'spec':spec,'one_minus_spec':1-spec,'cutoff':cutoffs}))
    points_1 = chart_1.mark_point(opacity=.5).encode(
                                        x=alt.X('one_minus_spec',axis=alt.Axis(title=["specificity","(% true negatives identified)"],
                                                                               values=[0,.2,.4,.6,.8,1],
                                                                               titleFontWeight=alt.FontWeight("bold"),
                                                                               labelExpr="(100-100*datum.label)")),
                                        y=alt.Y('sens',axis=alt.Axis(title=["sensitivity","(% true positives identified)"],
                                                                     values=[0,.2,.4,.6,.8,1],
                                                                     titleFontWeight=alt.FontWeight("bold"),
                                                                     labelExpr="(100*datum.label)")),
                                        tooltip=[alt.Tooltip('spec',format='.0%',title="specificity (\%)"),
                                                 alt.Tooltip('sens',format='.0%',title="sensitivity (\%)"),
                                                 alt.Tooltip('cutoff',format='.2f',title="classifier cut-off value")],
                                        size=alt.condition(~highlighter,alt.value(30),alt.value(60))
                                                ).add_selection(highlighter)
    line_1 = chart_1.mark_line(color='red',strokeDash=[5,3]).encode(x='sens',y='sens')
    altfig1 = points_1 + line_1
    altfig1 = altfig1.configure_axis(grid=False)
    fig1 = plt.figure()
    plt.plot([0,1],[0,1],'r--')
    plt.plot(1-spec,sens,linestyle='',marker='o',alpha=.4)
    plt.ylabel('sensitivity')
    plt.xlabel('specificity')
    plt.xticks([0,0.2,0.4,0.6,0.8,1],[1,.8,.6,.4,.2,0])
    #plt.savefig('figures\\' + 'area_under_curve_main.pdf', bbox_inches='tight',pad_inches=0.0)
    
    fig2 = plt.figure()
    plt.plot(cutoffs,spec,label="specificity")
    plt.plot(cutoffs,sens,label="sensitivity")
    plt.xlabel("cutoff value")
    plt.legend()
    
    TT = np.dot(xx,WW)
    Xz_complete = np.hstack((np.ones((len(TT[:,0]),1)),TT[:,0:ncomps]))
    logodds,yhat = logit_probs(Xz_complete,logit_res.params)
    
    status = np.array(["Not in sample"]*len(maindata))
    status[selecting] = "NO"
    status[selecting[yy.flatten()]] = "YES"
    insample = np.ones((n,)) == 0
    insample[selecting] = True
    df = pd.DataFrame({'First Principal Component':TT[:,0],
                                    'Second Principal Component':TT[:,1],
                                    'Predicted ' + attribute + '-ness':yhat,
                                    'logodds':logodds,
                                    'Occupation':list(maindata["Title"]),
                                    attr + "?":list(status),
                                    'In sample':list(insample)})
    altchart2 = alt.Chart(df,width=125,height=300)
    points21 = altchart2.mark_point().encode(color=alt.Color(attr + "?",
                                                                       scale=alt.Scale(domain=['YES','NO','Not in sample'],
                                                                       range=['green','red','lightgray'])),
                                             x=alt.X('First Principal Component'),
                                             y=alt.Y('Predicted ' + attribute + '-ness',axis=alt.Axis(values=[0,.2,.4,.6,.8,1],
                                                                     titleFontWeight=alt.FontWeight("bold")),
                                                     scale=alt.Scale(domain=(0,1))),
                                             tooltip=[alt.Tooltip('Occupation'),
                                                 alt.Tooltip('First Principal Component',title="1st Principal Component",format='.2f'),
                                                 alt.Tooltip('Predicted ' + attribute + '-ness',format='.2f')],
                                             opacity=alt.condition(~highlighter,alt.value(.6),alt.value(.975)),
                                             size=alt.condition(~highlighter,alt.value(30),alt.value(45))
                                               ).add_selection(highlighter).interactive()
    points22 = altchart2.mark_point().encode(color=alt.Color(attr + "?",
                                                                       scale=alt.Scale(domain=['YES','NO','Not in sample'],
                                                                       range=['green','red','lightgray'])),
                                             x=alt.X('Second Principal Component'),
                                             y=alt.Y('Predicted ' + attribute + '-ness',axis=alt.Axis(title = "",values=[0,.2,.4,.6,.8,1],
                                                                     titleFontWeight=alt.FontWeight("bold"),labelExpr=""),
                                                     scale=alt.Scale(domain=(0,1))),
                                             tooltip=[alt.Tooltip('Occupation'),
                                                 alt.Tooltip('Second Principal Component',title="2nd Principal Component",format='.2f'),
                                                 alt.Tooltip('Predicted ' + attribute + '-ness',format='.2f')],
                                             opacity=alt.condition(~highlighter,alt.value(.6),alt.value(.975)),
                                             size=alt.condition(~highlighter,alt.value(30),alt.value(45))
                                                ).add_selection(highlighter).interactive()
    altchart3 = alt.Chart(df,width=125,height=300)
    points31 = altchart3.mark_point().encode(color=alt.Color(attr + "?",
                                                                       scale=alt.Scale(domain=['YES','NO','Not in sample'],
                                                                       range=['green','red','lightgray'])),
                                             x=alt.X('First Principal Component'),
                                             y=alt.Y('logodds',title=['Predicted ' + attribute + '-ness','(log-odds scale)'],
                                                     axis=alt.Axis(labelExpr="format(1-1/(1+exp(datum.value)),'.2f')")),
                                             tooltip=[alt.Tooltip('Occupation'),
                                                 alt.Tooltip('First Principal Component',title="1st Principal Component",format='.2f'),
                                                 alt.Tooltip('Predicted ' + attribute + '-ness',format='.2f')],
                                             opacity=alt.condition(~highlighter,alt.value(.6),alt.value(.975)),
                                             size=alt.condition(~highlighter,alt.value(30),alt.value(45))
                                                ).add_selection(highlighter).interactive()
    points32 = altchart3.mark_point().encode(color=alt.Color(attr + "?",
                                                                       scale=alt.Scale(domain=['YES','NO','Not in sample'],
                                                                       range=['green','red','lightgray'])),
                                             x=alt.X('Second Principal Component'),
                                             y=alt.Y('logodds',axis=alt.Axis(title='',labelExpr="")),
                                             tooltip=[alt.Tooltip('Occupation'),
                                                 alt.Tooltip('Second Principal Component',title="2nd Principal Component",format='.2f'),
                                                 alt.Tooltip('Predicted ' + attribute + '-ness',format='.2f')],
                                             opacity=alt.condition(~highlighter,alt.value(.6),alt.value(.975)),
                                             size=alt.condition(~highlighter,alt.value(30),alt.value(45))
                                                ).add_selection(highlighter).interactive()
    st.session_state["step1_done"] = True
    st.session_state['altfig1'] = altfig1
    st.session_state['AUC'] = AUC
    st.session_state['altfig2'] = points21 | points22
    st.session_state['altfig3'] = points31 | points32
    st.session_state['yhat'] = yhat
    st.session_state['logitparams'] = logit_res.params


def run_analysis_true():
    st.session_state["step1_done"] = True


def resize_classification():
    st.session_state["resize_class"] = True


st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
        """,
    unsafe_allow_html=True
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)



if not "step1_done" in st.session_state:
    st.session_state['step1_done'] = False

if not "custom_input_received" in st.session_state:
    st.session_state["custom_input_received"] = False

if not "curclassification" in st.session_state:
    st.session_state["curclassification"] = {}

if not "createtime" in st.session_state:
    st.session_state["createtime"] = {}

if not "submissionstatus" in st.session_state:
    st.session_state["submissionstatus"] = {}

if not "user_defined_attributes" in st.session_state:
    st.session_state["user_defined_attributes"] = list([""])

if not "curpage" in st.session_state:
    st.session_state['curpage'] = 0



df_data,df_varinfo,XX,n,k,xx,vv,WW = data_load_and_PCA("occup_data_toanalyze.gzip","occup_varnames.gzip")

if not "randdraw" in st.session_state:
    st.session_state["randdraw"] = random_select(n)

#######
### Vertical layout

intro = st.container()
sample_controls = st.container()
classify_expander = st.expander("Classify random training sample")
classify_progress = st.container()
actionbuttons1 = st.columns([7,7,4])
results_holder = st.container()
contributeresults = results_holder.expander("Add to hive mind")
top5tables = results_holder.columns(2)
results_holder.markdown("---")
graphcols1 = results_holder.columns(2)
graphcols2 = results_holder.columns(2)
methodology = st.expander("Methodology + raw data")
###
#######

intro.title("Occupation classifier")
intro.write("""
This tool allows you to use machine learning to automatically classify occupations into different types. The information 
you provide by classifying a small subsample will extrapolated across 873 Department of Labor [O\*NET occupations](https://www.onetonline.org/ "O*NET Online website").
"""
)
instructions = intro.expander("Instructions")
instructions.write("""
To use the tool, first select an attribute from the dropdown menu, or add your own using the text entry box on the right. Then, classify a small, 
randomly-selected sample of occupations as either "Yes" or "No".
Finally, click the **Launch/Refresh Analysis** button below. The tool will use machine learning to extrapolate 
your choices onto the full set of 873 occupations.

After that, you can click the **Download results** button to download the classification scores you have generated.

You can also click on
the **Add to hive mind** button to contribute your classifications to an ongoing collaborative project to make robust classifications.
Providing your name and e-mail address is optional.
You can see the results of this project, including a list of who else has contributed, by clicking [this link](https://mattdelventhal.com/project/collaborative_classify/ "Collaborative classify").

Below the analysis results, you may click the **Test the model** button. This will launch an algorithm which uses a bootstrap method to 
evaluate how reliable the extrapolated results are likely to be.

Finally, an explanation of the methodology, along with downloadable raw data, can be found by clicking the **Methodology** tab at the very end of the page.
""")


methodology.write("""
This app was developed by [Matt Delventhal](https://mattdelventhal.com/ "Matt Delventhal's website"). It uses information provided by the U.S. Department of Labor's O*NET program, combined with judgments made by you, the user, 
to assign attributes to occupations. For example, we might ask: Are Audiologists an *Analytical* occupation? If so, how *Analytical* 
are they? What does *Analytical* mean in this context?

Our method sidesteps the problem of formal definition and assumes that you, the user, have a good intuitive sense of what *Analytical* 
(or some other attribute) means. All we need is for you to classify a small sample of occupations are analytical or not. Then we 
can infer a definition by comparing your choices to the rich set of hundreds of occupation characteristics provided by O*NET.

This is achieved by fitting a logistic regression to your classification choices using the first two principal components of an array 
of 353 O*NET occupation characteristics. This produces a score between 0 and 1. Strictly speaking, this represents the estimated probability that a 
given occupation will possess the given attribute. We can also think of this score as representing how *intensely* the occupation 
possesses that attribute (e.g. *how* Analytical is it?).

In what follows, we will first describe the selection of the initial set of characteristics and 
the results of the Principal Component Analysis. The application of the logistic regression is completely standard. (A useful explanation 
of the technique can be found [here](https://en.wikipedia.org/wiki/Logistic_regression "Logistic regression").)

##### O*NET Data

[O*NET Online](https://www.onetonline.org/ "O*NET Online") provides data on a wide range of occupation characteristics, which they have 
produced through a combination of surveys and expert analysis. This includes information on what kinds of skills are required to succeed 
in the occupation and what kinds of tasks workers in the occupation usually perform. This app uses data from the "Abilities," "Interests," 
"Knowledge," "Skills," "Work Activities," "Work Styles," and "Work Values" tables. These provide data for 873 occupations. Any variable which 
is missing for any of these occupations is excluded from our analysis. Some of the tables, such as the "Skills" table, provide information 
on both how important a skill is, and what level of skill is required. In those cases, both "importance" and "level" for that particular skill 
are registered as distinct variables. All variables are normalized linearly to take values between 0 and 1. The final set includes 353 
occupation characteristics.

You may download the dataset and the variable descriptions by clicking the buttons below."""
)
download_columns = methodology.columns(2)
download_columns[0].download_button(
        label = "Download dataset",
        data=convert_df(df_data),
        file_name='occup_data_toanalyze.csv',
        mime='text/csv',
        )
download_columns[1].download_button(
        label = "Download variable descriptions",
        data=convert_df(df_varinfo),
        file_name='occup_varnames.csv',
        mime='text/csv',
        )


methodology.write("""
The data is then arranged in an 873 by 353 matrix, where each column represents an occupation characteristic. The column-wise mean is 
subtracted from each column, so that each resulting column has a mean of 0. Then the eigenvectors of the resulting matrix are taken as 
the weight vectors defining the dataset's principal components, as is standard. 
[Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis "Principal Component Analysis") provides a useful description 
of this standard statistical technique.
""")
highlighter = alt.selection_single(on='mouseover')
screeplot = alt.Chart(pd.DataFrame({"component_number":[x+1 for x in list(range(len(vv[0:40])))],"variation_explained":vv[0:40]/np.sum(vv)}))
screedots = screeplot.mark_point().encode(y=alt.Y("variation_explained",axis=alt.Axis(format=".0%")),x="component_number",
            tooltip=[alt.Tooltip("component_number"),alt.Tooltip("variation_explained",format=".1%")],
            size=alt.condition(~highlighter,alt.value(20),alt.value(45))).add_selection(highlighter).interactive()
screelines = screeplot.mark_line().encode(y="variation_explained",x="component_number")


cumscreeplot = alt.Chart(pd.DataFrame({"component_number":[x+1 for x in list(range(len(vv[0:40])))],"cumulative_variation_explained":np.cumsum(vv[0:40]/np.sum(vv))}))
cumscreedots = cumscreeplot.mark_point().encode(y=alt.Y("cumulative_variation_explained",axis=alt.Axis(format=".0%")),x="component_number",
            tooltip=[alt.Tooltip("component_number"),alt.Tooltip("cumulative_variation_explained",format=".1%")],
            size=alt.condition(~highlighter,alt.value(20),alt.value(45))).add_selection(highlighter).interactive()
cumscreelines = cumscreeplot.mark_line().encode(y="cumulative_variation_explained",x="component_number")


screeoption = methodology.radio("Show:",['-','Variance explained','Cumulative variance explained'],index=1)
if screeoption == 'Variance explained':
    methodology.altair_chart((screedots + screelines))
elif screeoption == 'Cumulative variance explained':
    methodology.altair_chart((cumscreedots + cumscreelines))

methodology.write("""
The chart above shows the fraction of variance explained by each principal component. We use only the first two because experimentation 
showed that more than this can lead to severe over-fitting in small samples, while one component alone is often not enough to achieve 
a good classification.

We can get some idea of what each principal component represents by looking at the biggest weights for each. Click the boxes below
to show or hide the 
10 highest-weighted occupation characteristics for each of the first five components. As can be inferred from the previous chart, these 
components together account for almost two thirds of all the variation in the data.
""")

df_varinfo_copy = df_varinfo.copy()
if methodology.checkbox("Top ten weights for first component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,0])[::-1]
    df_varinfo_copy["PC_1_weights"] = WW[:,0]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_1_weights"]])

if methodology.checkbox("Top ten weights for second component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,1])[::-1]
    df_varinfo_copy["PC_2_weights"] = WW[:,1]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_2_weights"]])

if methodology.checkbox("Top ten weights for third component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,2])[::-1]
    df_varinfo_copy["PC_3_weights"] = WW[:,2]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_3_weights"]])

if methodology.checkbox("Top ten weights for fourth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,3])[::-1]
    df_varinfo_copy["PC_4_weights"] = WW[:,3]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_4_weights"]])

if methodology.checkbox("Top ten weights for fifth component"):
    firstcomploadings_sort = np.argsort(np.abs(WW)[:,4])[::-1]
    df_varinfo_copy["PC_5_weights"] = WW[:,4]
    methodology.write(df_varinfo_copy.loc[df_varinfo_copy.index[firstcomploadings_sort[0:10]]].reset_index()[["Name","Detail","PC_5_weights"]])



intro.write('*[Click here](https://mattdelventhal.com/project/collaborative_classify/ "Collaborative classify") to see a collaborative project based on the cumulative responses to this app.*')
intro.write('*An explanation of the methodology and downloadable raw data can be found by clicking the* ***Methodology*** *tab at the bottom of this page.*')
sample_size = sample_controls.slider("Sample Size",40,100,step=20,value=40)
st.session_state['curpage'] = min(st.session_state['curpage'],int(sample_size/10)-1)
    
#sample_size = st.select_slider("Sample Size",[40,60,80,100],value=40)


selected = np.argsort(st.session_state["randdraw"])[0:sample_size]
starting_attributes_list = ["Analytical","Prestigious","Dangerous","Manual"]


attribute_entry_cols = sample_controls.columns(2)

def input_response():
    st.session_state["custom_input_received"] = True

user_attribute = attribute_entry_cols[1].text_input("Add a new attribute to the list:",on_change=input_response)

if st.session_state["custom_input_received"]:
    if not user_attribute in st.session_state["user_defined_attributes"]:
        st.session_state["user_defined_attributes"] += [user_attribute]
        attribute_entry_cols[1].write(user_attribute + " added to the list.")
        if "" in st.session_state["user_defined_attributes"]:
            st.session_state["user_defined_attributes"].remove("")
    elif user_attribute in st.session_state["user_defined_attributes"]:
        attribute_entry_cols[1].write(user_attribute + " already in the list.")
    st.session_state["custom_input_received"] = False
else:
    attribute_entry_cols[1].write(" ")
    attribute_entry_cols[1].write(" ")

attribute = attribute_entry_cols[0].selectbox("Choose an attribute to classify:",starting_attributes_list +              
                                                st.session_state["user_defined_attributes"]) 

    



trainingsample = df_data.iloc[selected].reset_index()


demeaned_tomerge = pd.DataFrame(data=xx,columns=list(df_data.columns[3:]))
demeaned_tomerge["O*NET-SOC Code"] = df_data["O*NET-SOC Code"]
trainingsample = pd.merge(trainingsample[["O*NET-SOC Code","Title","Description"]],demeaned_tomerge,
                        how="left",on="O*NET-SOC Code")
trainingsample[attribute] = 0

if actionbuttons1[1].button("Draw new random sample"):
    del st.session_state["randdraw"]
    st.session_state["randdraw"] = random_select(n)
    selected = np.argsort(st.session_state["randdraw"])[0:sample_size]
    trainingsample = df_data.iloc[selected].reset_index()
    trainingsample = pd.merge(trainingsample[["O*NET-SOC Code","Title","Description"]],demeaned_tomerge,
                        how="left",on="O*NET-SOC Code")
    trainingsample[attribute] = 0
    for key in st.session_state["curclassification"]:
        st.session_state["curclassification"][key] = [-1]*n
    its = 0    
    for key in st.session_state["createtime"]:
        st.session_state["createtime"][key] = time.time() + its
        its += 1
    for key in st.session_state["submissionstatus"]:
        st.session_state["submissionstatus"][key] = ""

if not attribute in st.session_state["curclassification"]:
    st.session_state["curclassification"][attribute] = [-1]*n

if not attribute in st.session_state["createtime"]:
    st.session_state["createtime"][attribute] = time.time()

if not attribute in st.session_state["submissionstatus"]:
    st.session_state["submissionstatus"][attribute] = ""

page_options = ['-'] + [str(x+1) + " (" + str(int(x*10) + 1) + "-" + str(int((x+1)*10)) + ")" for x in range(int(sample_size/10))]
page_optdict = dict(zip(page_options,[0] + [x for x in range(int(sample_size/10))]))

with classify_expander:
    radiocontainer = st.container()
    input_field = st.container()
    navcols = st.columns(5)
    if navcols[0].button("Prev page",
                         disabled = ([True] + [False]*(len(page_options)-2)  )[st.session_state['curpage']]
                        ) and st.session_state['curpage'] > 0:
        st.session_state['curpage'] -= 1
    if navcols[4].button("Next page",
                         disabled = ([False]*(len(page_options)-2) + [True] )[st.session_state['curpage']]
                        ) and st.session_state['curpage'] < (len(page_options)-2):
        st.session_state['curpage'] += 1
    pageselect = page_optdict[radiocontainer.radio("Page:",page_options,index=st.session_state['curpage']+1)]
    st.session_state['curpage'] = pageselect
    with input_field:
        for j in range(int(pageselect*10),int((pageselect+1)*10)):
            curline = st.columns([1,17,4])
            curline[0].write("**" + str(j+1) + ".**")            
            curline[1].write("**" + trainingsample["Title"][j] + "**") #+ ": " + trainingsample["Description"][j])
            #curline[0].write("**" + trainingsample["Title"][j] + "**")
            if curline[1].checkbox("Detail",key=j):
                curline[1].write(trainingsample["Description"][j])
            curdefault = int(1 + st.session_state["curclassification"][attribute][j])
            curchoice = curline[2].radio(attribute + "?",["-","Yes","No"],key=j,index=curdefault)
            if curchoice == "Yes":
                st.session_state["curclassification"][attribute][j] = 0
            elif curchoice == "No":
                st.session_state["curclassification"][attribute][j] = 1
        
            
trainingsample[attribute] = np.array(st.session_state["curclassification"][attribute][0:sample_size])



ncomps = 2
num_classified = np.sum(np.array(st.session_state["curclassification"][attribute][0:sample_size]) != -1)
classify_progress.write("Training sample: " + str(num_classified) + " classified out of " + str(sample_size) + ".")
classify_bar = classify_progress.progress(0.0)
classify_bar.progress(num_classified/sample_size)
if num_classified == sample_size:
    analysis_ready = True
else:
    analysis_ready = False

if not analysis_ready:
    st.session_state["step1_done"] = False


actionbuttons1[0].button("Launch/Refresh Analysis",on_click=run_analysis,args=(trainingsample,attribute,sample_size,WW,xx,df_data,selected),disabled=not analysis_ready)
dl_buttonholder = actionbuttons1[2].empty()
dl_buttonholder.button("Download results",disabled=True)


#results_holder.button("Launch Analysis",on_click=run_analysis_true)

if st.session_state["step1_done"]:
    results_df = df_data[["O*NET-SOC Code","Title","Description"]].copy()
    results_df[attribute + "?"] = ""
    results_df.loc[selected,attribute + "?"] = "NO"
    results_df.loc[selected[np.array(trainingsample[attribute]) == 0],attribute + "?"] = "YES"
    results_df[attribute + "_score"] = np.array(st.session_state['yhat'])
    
    dl_buttonholder.download_button(
        label = "Download results",
        disabled=not st.session_state["step1_done"],
        data=convert_df(results_df),
        file_name=attribute + '_occu_classify.csv',
        mime='text/csv',
        )
    contributeresults.write('Contribute your classifications to [this collaborative project](https://mattdelventhal.com/project/collaborative_classify/ "Collaborative classify").')
    hivemindcols = contributeresults.columns([5,5,2])
    connectstatus = contributeresults.empty()
    connectstatus.write("**Status:**" + st.session_state["submissionstatus"][attribute])
    username = hivemindcols[0].text_input("Your name (optional)",value="")
    useremail = hivemindcols[1].text_input("Email address (optional)",value="")
    hivemindcols[2].write(" ")
    hivemindcols[2].write(" ")
    if hivemindcols[2].button("Send!"): # and not (st.session_state["submissionstatus"][attribute] == " Submitted successfully!"):
        subm_id = int(st.session_state["createtime"][attribute]*10)
        df_tosubmit = trainingsample[["O*NET-SOC Code",attribute]].copy()
        df_tosubmit["Onetcode"] = integer_onetcode(df_tosubmit,'O*NET-SOC Code')
        df_tosubmit[attribute] = 1*(np.array(trainingsample[attribute]) == 0)
        df_tosubmit["attribute"] = attribute
        df_tosubmit["submission_id"] = subm_id 
        connection = create_connection(sqlserver,sqlport, 
                               sqluser, sqlpwd,'occu_classify')        
        query = """
        REPLACE INTO submissions(submission_id,user_name,user_email,add_time)
        VALUES
        """
        
        query += "(" + str(subm_id) + ","
        query += '"' + username + '"' + ","
        query += '"' + useremail + '"' + ","
        query += '"' + str(datetime.datetime.utcfromtimestamp(st.session_state["createtime"][attribute])) + '"'
        query += ");"
        submission_record_success = execute_query_st(connection,query)
        
        query = """
        REPLACE INTO classify(submission_id,occup_id,attribute,user_classify)
        VALUES
        """
        query += df_to_sql_insert(df_tosubmit[['submission_id','Onetcode','attribute',attribute]],indexer=False) + ";"
        
        data_submission_success = execute_query_st(connection,query)
        connection.close()
        if submission_record_success and data_submission_success:
            st.session_state["submissionstatus"][attribute] = " Submitted successfully!"
            connectstatus.write("**Status:**" + st.session_state["submissionstatus"][attribute])
        else:
            st.session_state["submissionstatus"][attribute] = " Oops, something went wrong with the submission."
            connectstatus.write("**Status:**" + st.session_state["submissionstatus"][attribute])

    sortedoccupations = np.array(df_data["Title"])[np.argsort(st.session_state["yhat"])[::-1]]   
    
    top5tables[0].write(f'''
        ##### 5 most {attribute} occupations
        | | Occupation |
        | --- | ------------------ |
        | 1 | {sortedoccupations[0]} |
        | 2 | {sortedoccupations[1]} |
        | 3 | {sortedoccupations[2]} |
        | 4 | {sortedoccupations[3]} |
        | 5 | {sortedoccupations[4]} |
        ''')
    top5tables[1].write(f'''
        ##### 5 least {attribute} occupations
        | | Occupation |
        | --- | ------------------ |
        | {n-0} | {sortedoccupations[-1]} |
        | {n-1} | {sortedoccupations[-2]} |
        | {n-2} | {sortedoccupations[-3]} |
        | {n-3} | {sortedoccupations[-4]} |
        | {n-4} | {sortedoccupations[-5]} |
        ''')
    spacer = " "
    graphcols1[0].write(f"### {spacer} ") 
    graphcols1[0].write(" ") 
    graphcols1[0].write(" ") 
    graphcols1[0].write(" ") 
    graphcols1[0].write("Total area under the curve: {0:1.3f}".format(st.session_state["AUC"]))
    #results_holder.pyplot(st.session_state["fig1"])
    graphcols1[0].altair_chart(st.session_state["altfig1"].properties(width=320,height=320))
    graphcols2[0].markdown("##### Classification power")
    graphcols2[0].write(f"""
When choosing a cut-off score, such that all occupations above that score are "{attribute}", and all below it are not, there is a natural
trade-off between maximizing true positives (*sensitivity*) and avoiding false positives (*specificity*). A higher AUC indicates a more 
useful classifier, with more options for cut-off values to maximize either *sensitivity*, or *specificity*, or find a balance between the two.
An AUC of 1 indicates the model is a flawless classifier (for this sample), and an AUC of 0.5, represented by the dashed diagonal line, 
would indicate it has zero classification power.
""")
    #results_holder.pyplot(st.session_state["fig2"])
    #graphcols1[1].write("Total area under the curve: {0:1.3f}".format(st.session_state["AUC"]))
    graphoption = graphcols1[1].radio("Choose y-axis scale:",['-','linear','log odds'],index=1)
    if graphoption == 'linear':
        graphcols1[1].altair_chart(st.session_state["altfig2"])
    elif graphoption == 'log odds':
        graphcols1[1].altair_chart(st.session_state["altfig3"])
    graphcols2[1].write("##### Estimation results")
    graphcols2[1].write(f"""
The machine "learns" which occupations are more "{attribute}" by applying a Logit model to the first two principal components of an array
of more than 300 occupation characteristics. The two scatterplots show the relationship between each principal component and the estimated
"{attribute}" score. You can mouse-over each datapoint to see which occupation it represents. You may use the mouse wheel to zoom in and out.
""")    
    results_holder.write("---")
    if results_holder.button("Test the model"):
        ### Number of trials used to evaluate model
        num_trials = 200

        ### Size of randomly-selected training set used to evaluate model. Remainder of data will be used as test set in each trial.
        trainingset_size = int(sample_size/2)

        paramslist = []

        AUClist_insamp = []
        pseudoR2list_insamp = []
        LLnlist_insamp = []

        AUClist_outsamp = []
        pseudoR2list_outsamp = []
        LLnlist_outsamp = []
        
        results_holder.write("""Each trial divides the training sample in half, and sees how well the model can use the information in one half
 to classify the other half.""")
        num_trials = 200
        latest_it = results_holder.empty()
        bar = results_holder.progress(0.0)
        maxits = 20
        
        for j in range(num_trials):
            latest_it.write("Running trial " + str(j+1) + " of " + str(num_trials) + ".")
            bar.progress((j+1)/num_trials)
            
            its = 0
            while its < maxits:
                randdraw = [rand.random() for x in range(sample_size)]
                training = np.argsort(randdraw)[0:trainingset_size]
                out_of_sample = np.argsort(randdraw)[trainingset_size:]
                cur_trainsamp = trainingsample.iloc[training]
                cur_outofsamp = trainingsample.iloc[out_of_sample]
                if (np.sum(np.array(cur_trainsamp[attribute]) == 0) > 1)\
                        and (np.sum(1 - (np.array(cur_trainsamp[attribute]) == 0)) > 1)\
                        and (np.sum(np.array(cur_outofsamp[attribute]) == 0) > 1)\
                        and (np.sum(1 - (np.array(cur_outofsamp[attribute]) == 0)) > 1):
                    break
                its += 1
            
            NN_train = len(cur_trainsamp)
            yy_train = np.reshape(np.array(cur_trainsamp[attribute]) == 0,(NN_train,1))
            xx_t_train = np.array(cur_trainsamp.drop(columns=["O*NET-SOC Code",attribute,"Title","Description"]))
            TT_t_train = np.dot(xx_t_train,WW)
            
            Xz_train = np.hstack((np.ones((NN_train,1)),TT_t_train[:,0:ncomps]))
            
            NN_out = len(cur_outofsamp)
            yy_out = np.reshape(np.array(cur_outofsamp[attribute]) == 0,(NN_out,1))
            xx_t_out = np.array(cur_outofsamp.drop(columns=["O*NET-SOC Code",attribute,"Title","Description"]))
            TT_t_out = np.dot(xx_t_out,WW)
            Xz_out = np.hstack((np.ones((NN_out,1)),TT_t_out[:,0:ncomps]))

            logit_res_train = robust_logitter(Xz_train,yy_train,disp=0)
            
            paramslist.append(logit_res_train.params)
            
            logit_yhat,yy_hat,LLn_insamp,pseudoR2_insamp,AUC_insamp,sens,spec,cutoffs=\
                    logit_fit(Xz_train,yy_train,logit_res_train.params)
            AUClist_insamp.append(AUC_insamp)
            pseudoR2list_insamp.append(pseudoR2_insamp)
            LLnlist_insamp.append(LLn_insamp)
            
            logit_yhat,yy_hat,LLn_outsamp,pseudoR2_outsamp,AUC_outsamp,sens,spec,cutoffs=\
                    logit_fit(Xz_out,yy_out,logit_res_train.params)
            AUClist_outsamp.append(AUC_outsamp)
            pseudoR2list_outsamp.append(pseudoR2_outsamp)
            LLnlist_outsamp.append(LLn_outsamp)
            
        testingchart = alt.Chart(pd.DataFrame({"Area_Under_The_Curve":AUClist_insamp + AUClist_outsamp,
                                               "Sample":["In-sample/training set"]*num_trials + ["Out-of-sample/testing set"]*num_trials
                                                }),
                                 width=640)
        bars = testingchart.mark_bar(opacity=0.3,binSpacing=0).encode(
            x=alt.X("Area_Under_The_Curve:Q", bin=alt.Bin(maxbins=50),scale=alt.Scale(domain=(.975*np.min(np.array(AUClist_outsamp)),1)),
                    axis=alt.Axis(labelAngle = 45,title="Area Under the Curve")),
            y=alt.Y('count()',stack=None,title="Num. of trials"),
            color=alt.Color("Sample",legend=alt.Legend(title="")),
        )
        
        
        testingchart2 = alt.Chart(pd.DataFrame({"themean":[np.mean(np.array(AUClist_insamp)),np.mean(np.array(AUClist_outsamp))],
                                                "color":["#1f77b4","#ff7f0e"],
                                                "Mean AUC":['In-sample: {0:1.3f}'.format(np.mean(np.array(AUClist_insamp))),'Out-of-sample: {0:1.3f}'.format(np.mean(np.array(AUClist_outsamp)))],
                                                })
        )        
        vrule = testingchart2.mark_rule().encode(x = "themean",color=alt.Color("color",
                                                                       scale=None),strokeDash=alt.StrokeDash("Mean AUC",scale=alt.Scale(
                                                                                    domain=['In-sample: {0:1.3f}'.format(np.mean(np.array(AUClist_insamp))),'Out-of-sample: {0:1.3f}'.format(np.mean(np.array(AUClist_outsamp)))],
                                                                                    range=[[5,3],[5,3]])
                                                                                    )
                                                )
        results_holder.altair_chart(bars + vrule)
        results_holder.write("##### Model validation")
        results_holder.write(f"""
The chart above shows the distribution of AUC across the training sets and testing sets for {str(num_trials)} trials. If the average in-sample AUC
is high enough, and the average out-of-sample AUC is not too much lower, this indicates that the model is likely to perform well when applied 
to the broader set of occupations. In that case, we should have confidence that the "{attribute}" scores generated by the model are meaningful.
""")

