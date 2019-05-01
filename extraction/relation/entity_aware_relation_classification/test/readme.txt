In the common test source code, it implemented the abstract parent class which is common parent class, but I do have any idea to instantiate the abstract parent class. 
Thus, I modified test code to run unit test, but I kept the test logic.
I replaced abstract parent class to my child class. Moreover, I modified test data because the entity label is heterogeneous. I trained each three common dataset seperately, and each dataset has different entity label, so, it causes error when the input data is mixed with these three common dataset. 

Therefore, I modified my unit test source code and provides correct test data for the unit test.