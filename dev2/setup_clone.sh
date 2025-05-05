#!/bin/bash

# Update package lists
sudo apt-get update

# Install required packages
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    openmpi-bin \
    libopenmpi-dev \
    metis

# Install Python packages
pip3 install --user mpi4py networkx matplotlib scipy

# Create project directory
mkdir -p /home/fatima/Desktop/PDC_Project/dev2

# Set up SSH for MPI
echo "Setting up SSH for MPI..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

echo "Setup completed!" 