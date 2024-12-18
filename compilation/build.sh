#!/bin/bash

cmake -S . -B build -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12
cmake --build build

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Python site-packages directory: $SITE_PACKAGES"

# Check if .so file exists (replace 'your_module.so' with your actual .so file name)
if [ ! -f "build/libdeep.so" ]; then
    echo "Error: .so file not found in current directory"
    exit 1
fi

# Copy the .so file to site-packages
# Using sudo because system-wide site-packages usually needs root permissions
echo "Copying .so file to $SITE_PACKAGES"
sudo cp build/libdeep.so "$SITE_PACKAGES"

# Verify the copy
if [ -f "$SITE_PACKAGES/libdeep.so" ]; then
    echo "Successfully installed module"
    # Set proper permissions
    sudo chmod 755 "$SITE_PACKAGES/libdeep.so"
else
    echo "Failed to install module"
    exit 1
fi

