  
#!/usr/bin/env bash
echo "Run test suites! "
coverage run --source=nsmr -m unittest discover tests # generate coverage file