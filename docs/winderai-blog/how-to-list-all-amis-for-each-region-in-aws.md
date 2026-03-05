# 如何在 AWS 中列出每个地区的所有 ami

> 原文：<https://winder.ai/how-to-list-all-amis-for-each-region-in-aws/>

当前的一个项目需要所有地区的 Amazon 机器映像(AMIs)列表，以便在 terraform 中使用。我找不到一个脚本来帮我做这件事，所以在这里你会找到一个使用 aws cli、jq 和一点 Bash 的脚本。

要使用它，只需在每个区域传递您想要查找的图像的全名。这必须是唯一的，否则它将只选择第一个 AMI。例如:

```py
$ ./images.sh RHEL-7.4_HVM_GA-20170808-x86_64-2-Hourly2-GP2
"ap-south-1" = "ami-e41b618b"
"eu-west-3" = "ami-39902744"
"eu-west-2" = "ami-a1f5e4c5"
"eu-west-1" = "ami-bb9a6bc2"
"ap-northeast-2" = "ami-0f5a8361"
"ap-northeast-1" = "ami-30ef0556"
"sa-east-1" = "ami-a789ffcb"
"ca-central-1" = "ami-dad866be"
"ap-southeast-1" = "ami-10bb2373"
"ap-southeast-2" = "ami-ccecf5af"
"eu-central-1" = "ami-d74be5b8"
"us-east-1" = "ami-c998b6b2"
"us-east-2" = "ami-cfdafaaa"
"us-west-1" = "ami-66eec506"
"us-west-2" = "ami-9fa343e7" 
```

脚本如下:

```py
#!/bin/bash if [ -z "$1" ] ; then
    echo "Please pass the name of the AMI"
    exit 1
fi

IMAGE_FILTER="${1}"

declare -a REGIONS=($(aws ec2 describe-regions --output json | jq '.Regions[].RegionName' | tr "\\n" " " | tr -d "\\"")) for r in "${REGIONS[@]}" ; do
 ami=$(aws ec2 describe-images --query 'Images[*].[ImageId]' --filters "Name=name,Values=${IMAGE_FILTER}" --region ${r} --output json | jq '.[0][0]')  printf "\\"${r}\\" = ${ami}\\n" done 
```