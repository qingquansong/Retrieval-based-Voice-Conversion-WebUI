#!/bin/bash

#copy my key pairs and configs from my network drive
cp /workspace/.ssh/* /root/.ssh/
#lower the private key access permission
chmod 600 /root/.ssh/github

#register the private keys
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/github

#set user
git config --global user.email "cc.chaocui@gmail.com"

#auto setup origin
git config --global push.autoSetupRemote true
