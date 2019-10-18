import pandas as pd
from sklearn.utils import resample


data = pd.read_csv("Mdata.csv", encoding="utf8", dtype='unicode')
columns = [	'Activity_Id',	'Assigned_To',	'Activity_Status',	'SR_Number',	'SR_Status',	'Activity_Sub_Status',	'SR_Cause_Code',	'SR_Symptom_Code',	'Asset_Decision_Code',	'Actual_Duration',	'SR_Customer_Description',	'res']

dataframe = pd.DataFrame(data, columns = [	'Activity_Id',	'Assigned_To',	'Activity_Status',	'SR_Number',	'SR_Status',	'Activity_Sub_Status',	'SR_Cause_Code',	'SR_Symptom_Code',	'Asset_Decision_Code',	'Actual_Duration',	'SR_Customer_Description',	'res'])

df = dataframe.drop(dataframe.index[0])
#print(df.head(5))
df['res'].replace(to_replace='Inbound Call', value='0' , inplace=True)
df.replace(to_replace='Help Desk', value='1' , inplace=True)
df.replace(to_replace='Field Repair', value='2', inplace=True)

print(df['res'])

df_majority1 = df['res']==2
df_majority2 = df['res']==0
df_minority= df['res']==1


# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=4514, random_state=3)
df_upsampled = pd.concat([df_minority, df_minority_upsampled])

df_majority1_downsample = resample(df_majority1, replace=False, n_samples=4514, random_state=3)
df_downsample1 = pd.concat([df_majority1, df_majority1_downsample])

df_majority2_downsample = resample(df_majority2, replace=False, n_samples=4514, random_state=3)
df_downsample2 = pd.concat([df_majority2, df_majority2_downsample])
res= pd.concat([df_downsample1, df_downsample2, df_upsampled])
print(res)

print(df_downsample1.value_counts())
print(df_downsample2.value_counts())
print(df_upsampled.value_counts())

#print(df_minority_upsampled)


