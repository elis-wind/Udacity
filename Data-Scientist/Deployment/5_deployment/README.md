# Instructions Deploying from the Classroom

Here is the code used in the screencast to get the web app running:

- Create a new folder web_app, and move all of the application folders and files to the new folder:  

`cd 5_deployment/`  
`mkdir web_app`  
`mv -t web_app/ data/ worldbankapp/ worldbank.py  wrangling_scripts/ requirements.txt runtime.txt`

- [Applicable only for the Local practice. Not for the workspace.] Create a virtual environment and then activate the environment:

`# Update Python`  
`conda update python`  
`# Run the following from the Exercise folder`  
`# Create a virtual environment`  
`python3 -m venv worldbankvenv`  
`# Activate the new environment (Mac/Linux)`  
`source worldbankenv/bin/activate`

The new environment will automatically come with Python packages meant for data science. In addition, pip install the specific Python packages needed for the web app

`pip install flask==0.12.5 pandas==0.23.3 plotly==2.0.15 gunicorn==19.10.0`

- Install the Heroku command-line tools. The classroom workspace already has Heroku installed.

`# Verify the installation`  
`heroku --version`  
`# Install, if Heroku not present`  
`curl https://cli-assets.heroku.com/install-ubuntu.sh | sh`

For your local installation, you can refer to the official installation instructions. And then log into heroku with the following command

`heroku login -i`

Heroku asks for your account email address and password, which you type into the terminal and press enter.

- The next steps involve some housekeeping:

    remove `app.run()` from worldbank.py

    type `cd web_app` into the Terminal so that you are inside the folder with your web app code.

Then create a proc file, which tells Heroku what to do when starting your web app:

`touch Procfile`

Then open the Procfile and type:

`web gunicorn worldbank:app`

- [Applicable only for the Local practice. Not for the workspace.] Create a requirements.txt file, which lists all of the Python packages that your app depends on:

`pip freeze > requirements.txt`

For workspace users, the requirements.txt is already available in the exercise folder. In addition, we have also provided a runtime.txt file in the exercise folder, that declares the exact Python version number to use. Heroku supports [these](https://devcenter.heroku.com/articles/python-support#supported-runtimes) Python runtimes.

- Initialize a git repository and make a commit:

`# Run it just once, in the beginning`
`git init`
`# For the first time commit, you need to configure the git username and email:`  
`git config --global user.email "you@example.com"`  
`git config --global user.name "Your Name"`

Whenever you make any changes to your web_app folder contents, you will have to run `git add` and `git commit` commands.

`# Every time you make any edits to any file in the web_app folder`  
`git add .`  
`# Check which files are ready to be committed`  
`git status`  
`git commit -m "your message"`

- Now, create a Heroku app:

`heroku create my-app-name --buildpack heroku/python`  
`# For example,`  
`# heroku create sudkul-web-app --buildpack heroku/python`  
`# The output will be like:`  
`# https://sudkul-web-app.herokuapp.com/ | https://git.heroku.com/sudkul-web-app.git`

where my-app-name is a unique name that nobody else on Heroku has already used. You can optionally define the build environment using the option `--buildpack heroku/python` The `heroku create` command should create a git repository on Heroku and a web address for accessing your web app. You can check that a remote repository was added to your git repository with the following terminal command:

`git remote -v`

- Before you finally push your local git repository to the remote Heroku repository, you will need the following environment variables (kind of secrets) to send along:

`# Set any environment variable to pass along with the push`  
`heroku config:set SLUGIFY_USES_TEXT_UNIDECODE=yes`  
`heroku config:set AIRFLOW_GPL_UNIDECODE=yes`  
`# Verify the variables`  
`heroku config`

If your code uses any confidential variable value, you can use this approach to send those values secretly. These values will not be visible to the public users. Now, push your local repo to the remote Heroku repo:

`# Syntax`  
`# git push <remote branch name> <local branch name>`  
`git push heroku master`

Other useful commands are:

`# Clear the build cache`  
`heroku plugins:install heroku-builds`  
`heroku builds:cache:purge -a <app-name> --confirm <app-name>`  
`# Permanently delete the app`  
`heroku apps:destroy  <app-name> --confirm <app-name>`
