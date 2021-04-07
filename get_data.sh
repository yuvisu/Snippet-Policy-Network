####################################
#   GET ICBEB 2018 DATABASE
####################################
mkdir -p tmp_data
cd tmp_data
wget http://2018.icbeb.org/file/REFERENCE.csv
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet1.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet2.zip
wget http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip
unzip TrainingSet1.zip
unzip TrainingSet2.zip
unzip TrainingSet3.zip
cd ..
python util/convert_ICBEB.py
