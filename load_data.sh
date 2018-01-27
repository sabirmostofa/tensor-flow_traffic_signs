# download Training zip file
curl http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip > GTSRB_Training.zip


# unzip Training zip file 
unzip GTSRB_Training.zip 

# download Test zip file
curl http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip > GTSRB_Test.zip


# unzip Test zip file 
unzip GTSRB_Test.zip

#delete zip file
rm GTSRB_Training.zip
rm GTSRB_Test.zip
