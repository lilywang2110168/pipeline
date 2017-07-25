import gender_guesser.detector as gender
from string import digits


d = gender.Detector(case_sensitive=False)
FEMALE = 0;
MALE = 1;
UNKNOWN= 2;



def guessGender(fullname):

	fullname = fullname.replace('"', ' ')
	fullname=fullname.translate(None, digits)
	name = fullname.split()
	#if the len is bigger than 1
	if len(name) > 0:
		##checks the first name
		gender=codeGender(d.get_gender(name[0]))
		if gender!=UNKNOWN:
			return gender

		else:
			for item in name:

				if item.lower()=='mom' or item.lower()=='girl' or item.lower()=="ms." or item.lower()=='miss' or item.lower()=='mrs.' item.lower()=='lady':
					return FEMALE
				if item.lower()=='dad' or item.lower()=='daddy' or item.lower()=="mr." or item.lower()=='boy':
					return MALE
	
			return UNKNOWN
		


def codeGender(gender):
	if gender=='female' or gender=='mostly_female':
		return FEMALE
	elif gender=='male' or gender=='mostly_male':
		return MALE
	else:
		return UNKNOWN



##main starts here
with open('../resources/fullnames.txt') as f:
	for line in f:
		fullname = line.strip()
		print (guessGender(fullname))








    
                   
