wget https://uofi.box.com/shared/static/vi16hr2ifklnzgmzktpv1e0a0s3pm05s -O train_gt.zip && 
unzip -d ./data train_gt.zip && 
rm train_gt.zip &&
wget https://uofi.box.com/shared/static/7aa6ba1tf0jpjg5c1nwzbifypbol4psh -O val_gt.zip && 
unzip -d ./data val_gt.zip &&
rm val_gt.zip

