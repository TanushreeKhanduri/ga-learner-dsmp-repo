# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path, delimiter=",", skip_header=1)
print(data.astype(int))
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]

#Code starts here
#census = np.concatenate(data,np.array(new_record))
census = np.concatenate((data,np.array(new_record)))


# --------------
#Code starts here
age = np.array(census[...,0])
max_age = np.max(age)
min_age = np.min(age)
age_mean = age.mean()
age_std = np.std(age)

print(max_age,min_age,age_mean,age_std)


# --------------
#Code starts here
race_0 = census[np.where(census[...,2] == 0)]
race_1 = census[np.where(census[...,2] == 1)]
race_2 = census[np.where(census[...,2] == 2)]
race_3 = census[np.where(census[...,2] == 3)]
race_4 = census[np.where(census[...,2] == 4)]

len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
length_array =  np.array([len_0,len_1,len_2,len_3,len_4])
minimum_length = min(length_array)
minority_race_array, = np.where(length_array == minimum_length)
minority_race =  minority_race_array[0]


# --------------
#Code starts here
senior_citizens = census[np.where(census[...,0] > 60)]
working_hours_sum = np.sum(senior_citizens[:,6])
senior_citizens_len = len(senior_citizens)
avg_working_hours = working_hours_sum/senior_citizens_len
print(avg_working_hours)


# --------------
#Code starts here
high = census[np.where(census[...,1] > 10)]
low = census[np.where(census[...,1] <= 10)]
avg_pay_high = np.sum(high[:,7])/len(high)
avg_pay_low = np.sum(low[:,7])/len(low)
if(avg_pay_high > avg_pay_low):
    print('yes')
else:
    print('no')


