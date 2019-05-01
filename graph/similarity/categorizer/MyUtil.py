# -*- coding: ms949 -*-

import os
import random



def divideList(user_list, group_number):




	if group_number > len(user_list):
		group_number = len(user_list)

	ret = []
	for i in range(group_number):
		ret.append( [] )

	indexes = range(len(user_list))
	random.shuffle(indexes)

	g = 0

	for i in range( len(indexes)):
		real_index = indexes[i]
		ret[ g ].append(  user_list [real_index ]  )

		g += 1
		if g == group_number:
			g = 0


	return ret


# -----------------------------------------------------------------
def sortDict( user_dict, descending_order = False ):

	return sorted(user_dict, key=user_dict.get, reverse=descending_order)


def sortDictHavingValue(user_dict, descending_order=False):


	r = []
	a = sorted( user_dict.items(), key=lambda x:x[1], reverse=descending_order)
	for key, values in a:
		r.append(key)
	return r


def sortDictHavingList(user_dict, index_in_list, descending_order = False):



	r = []
	a = sorted( user_dict.items(), key=lambda x:x[1][index_in_list], reverse=descending_order)
	for key, values in a:
		r.append(key)
	return r





def getMyFileName(user_file):
	return os.path.basename(user_file)

def getMyPath(user_path):
	f = getMyFileName(user_path)
	return os.path.abspath(user_path).replace(f,'')




def getRandomString(length):
	str1 = 'abcdefghijklmnopqrstuvwxyz'
	str2 = str1.upper()
	numbers = '0123456789'

	box = str1+str2 + numbers

	r = ''

	for i in range(length):
		s = random.sample(box, 1) [0]
		r += s
	return r






