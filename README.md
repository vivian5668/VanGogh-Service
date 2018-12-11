# AI Painting Van Gogh - Services
### see Front-end here:
https://github.com/vivian5668/VanGogh-Client

---
This is not a complete repository for all micro-services for the AI Painting Van Gogh project.
All code lives on AWS. The main puspose of this repository is to save important milestone progress
for the main Convolutional Neural Network model implemented by Pytorch and some local testing. 

### Application data flow
user upload -> API Gateway -> Trigger AWS Step Function

### AWS Step Function Work Flow
lambda -> checks DynamoDB check and write UUID -> trigger FarGate -> write result 
to S3 -> SNS is triggered once new image is in S3 -> send result image to user


reference
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

- to test docker locally, make sure AWS credentials are in `~/.aws/credentials`
- `bash sample_docker_run.sh`
