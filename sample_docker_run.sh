awsaccesskey=$(awk "NR==2" ~/.aws/credentials | sed 's/.*= \(.*\)/\1/g')
awsaccessscrete=$(awk "NR==3" ~/.aws/credentials | sed 's/.*= \(.*\)/\1/g')
docker run -t -e CONTENT_IMAGE_PATH="s3://chelsea-style-transfer/content/094f6639-ccf7-4ba9-84d2-c5ff8adf4252.jpg" \
-e STYLE_IMAGE_PATH="s3://chelsea-style-transfer/Van_style.jpg" \
-e OUTPUT_IMAGE_PATH="s3://chelsea-style-transfer/test.jpg" \
-e RESOLUTION="200" \
-e AWS_ACCESS_KEY_ID=${awsaccesskey} \
-e AWS_SECRET_ACCESS_KEY=${awsaccessscrete} \
style-transfer:latest
