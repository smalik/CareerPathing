'''
Codelplan:
	1. connect to mysql
	2. pull tables as pandas datasets
	3. Identify and merge descriptive name of element_id into tables
	4. For 2-Dimensional tables, simply apply pivot table as in 4.a separately for each metric we want added (e.g. data value, standard deviation, etc...).  SOC code is always the primary and only key.
		4.a Concatenate element_id descriptor label with scale id and perform pivot
		pd.pivot_table(df_mysql, values= 'data_value', index= ['onetsoc_code'], columns= ['element_id', 'scale_id'], aggfunc= np.sum) 
       4.b Tables: 
          i.      abilities
          ii.     knowledge
          iii.    skills
          iv.     work_styles
          v.      interests
          vi.     work values
          vii.    work activities
	5. For n-Dimension tables, apply pivot_table as in 5.a, and be sure to fill in missing values with 0.  Compute the margins in pandas as part of the 
		5.a Concatenate primary id field with the relavent dimensions, but remember to always handle the multiple levels separately and then merge them together with the master (exhaustive) SOC list.  Be sure not to drop rows in the final table.
			pd.pivot_table(df_mysql, values= 'data_value', index= ['onetsoc_code'], columns= ['category'], aggfunc= np.mean, dropna= True, fill_value = 0)
      5.b Tables:
          i.      education_training_experience
          ii.     task_ratings
          iii.    work context
	6. On final table, add new columns that break down the SOC hierarchy

    10. Direct Match title file: http://www.bls.gov/soc/soc_2010_direct_match_title_file.xls
'''


import pymysql
import pandas as pd
import pandas.io.sql as psql
import numpy as np

def getOnetPanelPivot(dsn):
    dsn['element_id'] = dsn['element_id'].map(lambda s: s.replace(",","").replace("-","").replace(" ","_"))+ "_" +dsn['scale_id']
    dsn.tr = pd.pivot_table(dsn, index=dsn.index, columns=['element_id'], values=['data_value'])
    cNames = pd.Index([e[1] for e in dsn.tr.columns.tolist()])
    dsn.tr.columns = cNames
    return dsn.tr

def getOnetPanelPivot_taskrating(dsn):
    dsn['task_id'] = dsn['task_id'].join('_').dsn['scale_id'] + '_' + dsn['category']
    dsn.tr = pd.pivot_table(dsn, index=dsn.index, columns=['task_id'], values=['data_value'])
    cNames = pd.Index([e[1] for e in dsn.tr.columns.tolist()])
    dsn.tr.columns = cNames
    return dsn.tr



# from read_frame.  ?datetime objects returned?  convert to datetime64?
def read_db(sql, con):
    return psql.frame_query(sql, con)

def db_colname(pandas_colname):
    '''convert pandas column name to a DBMS column name
       TODO: deal with name length restrictions, esp for Oracle
    '''
    colname =  pandas_colname.replace(' ','_').strip()
    return colname
        
def split_soccode(df):
    '''
        Break out the SOC code by classification hierarchy
    '''
    soc = df.index
    df['soc_major'] = soc.map(lambda s: s.split('-')[0])
    df['soc_minor'] = soc.map(lambda s: s.split('-')[1][0:2])
    df['soc_broad'] = soc.map(lambda s: s.split('-')[1][2:4])
    df['soc_occ'] = soc.map(lambda s: s.split('.')[0])

    print(df.head())

    return df

def transform_ONET_tbl_1D(db, tbl_ONET, comp_measure = np.mean, stdErr= True):
    #sql_str = 'select * from ' + tbl_ONET
    if(stdErr):
      sql_str =   'select a.onetsoc_code, b.element_name, c.scale_name, a.element_id, a.scale_id, \
                  CONCAT(b.element_name, "_", a.scale_id) as col_name, a.data_value, a.standard_error, a.domain_source from ' + tbl_ONET + \
                  ' a join content_model_reference b using(element_id) join scales_reference c using(scale_id)'
    elif(not stdErr):
      sql_str =   'select a.onetsoc_code, b.element_name, c.scale_name, a.element_id, a.scale_id, \
                  CONCAT(b.element_name, "_", a.scale_id) as col_name, a.data_value, a.domain_source from ' + tbl_ONET + \
                  ' a join content_model_reference b using(element_id) join scales_reference c using(scale_id)'

    print('Transforming ' + tbl_ONET + ':') 

    df = pd.read_sql(sql_str, con=db)
    print('loaded dataframe from MySQL. records:', len(df))

    ##  NEED TO MERGE IN THE DESCRIPTIVE NAME HEADERS FROM OTHER TABLES TO DESCRIBE ELEMENT NAME
    dsn = pd.pivot_table(df, values= 'data_value', index= ['onetsoc_code'], columns= ['element_name', 'scale_name'], aggfunc= comp_measure, fill_value= 0)

    return dsn

def transform_ONET_tbl_2D(db, tbl_ONET, comp_measure = np.mean, stdErr= True):
    if(stdErr):
      sql_str =   'select a.onetsoc_code, b.element_name, c.scale_name, a.element_id, a.scale_id, a.category, \
                  CONCAT(b.element_name, "_", a.scale_id, "_", a.category) as col_name, a.data_value, a.standard_error, a.domain_source from ' + tbl_ONET + \
                  ' a join content_model_reference b using(element_id) join scales_reference c using(scale_id)'
    elif(not stdErr):
      sql_str =   'select a.onetsoc_code, b.element_name, c.scale_name, a.element_id, a.scale_id, a.category, \
                  CONCAT(b.element_name, "_", a.scale_id, "_", a.category) as col_name, a.data_value,  a.domain_source from ' + tbl_ONET + \
                  ' a join content_model_reference b using(element_id) join scales_reference c using(scale_id)'

    print('Transforming ' + tbl_ONET + ':') 

    df = pd.read_sql(sql_str, con=db)
    print('loaded dataframe from MySQL. records:', len(df))

    dsn = pd.pivot_table(df, values= 'data_value', index= ['onetsoc_code'], columns= ['col_name'], aggfunc= comp_measure, fill_value= 0)
    split_soccode(dsn)

    return dsn

def transform_ONET_trTbl(db, tbl_ONET, comp_measure = np.mean):
    sql_str =   'select a.onetsoc_code, a.task_id, b.task, a.scale_id, a.category, CONCAT(b.task_id, "_", a.scale_id, "_", a.category) as col_name, a.data_value, \
                a.standard_error, a.domain_source from ' + tbl_ONET + ' a join task_statements b using(onetsoc_code, task_id) join scales_reference c using(scale_id)'

    print('Transforming ' + tbl_ONET + ':') 

    df = pd.read_sql(sql_str, con=db)
    print('loaded dataframe from MySQL. records:', len(df))

    dsn = pd.pivot_table(df, values= 'data_value', index= ['onetsoc_code'], columns= ['col_name'], aggfunc= comp_measure, fill_value= 0)
    split_soccode(dsn)

    return dsn



db = pymysql.connect(host="dev.plytos.com", user="plytos", passwd="plytos1", db="onet")

# Merge tables into one large table.
# Merge on SOC Code as key
abilities = transform_ONET_tbl_1D(db=db, tbl_ONET='abilities')
knowledge = transform_ONET_tbl_1D(db=db, tbl_ONET='knowledge')
interests = transform_ONET_tbl_1D(db=db, tbl_ONET='interests', stdErr= False)
skills    = transform_ONET_tbl_1D(db=db, tbl_ONET='skills')
work_values     = transform_ONET_tbl_1D(db=db, tbl_ONET='work_values', stdErr= False)
work_styles     = transform_ONET_tbl_1D(db=db, tbl_ONET='work_styles')
work_activities = transform_ONET_tbl_1D(db=db, tbl_ONET='work_activities')

ete = transform_ONET_tbl_2D(db=db, tbl_ONET='education_training_experience')
work_context = transform_ONET_tbl_2D(db=db, tbl_ONET='work_context')
task_ratings = transform_ONET_trTbl(db=db, tbl_ONET='task_ratings')

new = pd.merge(abilities, knowledge, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(abilities, knowledge, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, interests, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, skills, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, work_values, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, work_styles, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, work_activities, left_index= True, right_index= True, how='outer')
print(new.shape)
new.columns = ['_'.join(col).strip() for col in new.columns.values]

new = pd.merge(new, work_context, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, ete, left_index= True, right_index= True, how='outer')
print(new.shape)
new = pd.merge(new, task_ratings, left_index= True, right_index= True, how='outer')
print(new.shape)
print(new.head())
print(new.columns)

new.to_csv('new.csv')

#dsn = pd.pivot_table(df_mysql, values= 'data_value', index= ['onetsoc_code'], columns= ['element_id', 'scale_id'], aggfunc= np.sum) 
#dsn = pd.pivot_table(df_mysql, values= 'data_value', index= ['onetsoc_code'], columns= ['task_id'], aggfunc= np.mean, dropna= False, fill_value = 0)
##dsn = pd.pivot_table(df_mysql, values= 'data_value', index= ['onetsoc_code'], columns= ['labels'], aggfunc= np.mean, dropna= False, fill_value = 0) 
db.close()
