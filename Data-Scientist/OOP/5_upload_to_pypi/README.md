# Putting Code on PyPi

## PyPi vs. Test PyPi

Note that pypi.org and test.pypy.org are two different websites. You'll need to register separately at each website. If you only register at pypi.org, you will not be able to upload to the test.pypy.org repository.

Also, remember that your package name must be unique. If you use a package name that is already taken, you will get an error when trying to upload the package.

## Summary of the Terminal Commands

`python setup.py sdist`

`pip install twine`

**commands to upload to the pypi test repository**

`twine upload --repository-url`

`https://test.pypi.org/legacy/ dist/*`

`pip install --index-url https://test.pypi.org/simple/ distributions`


**command to upload to the pypi repository**

`twine upload dist/*`

`pip install distributions`
