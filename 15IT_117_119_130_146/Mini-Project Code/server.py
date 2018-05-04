from flask import Flask, render_template
from build_tags import suggestionsList

app = Flask(__name__)


def replacePlus(s):
    return '+'.join(s.split())


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/<movie_id>')
def suggest(movie_id):
    movie_name, suggestions = suggestionsList(int(movie_id))
    return render_template('show.html',
                           movie_name=movie_name,
                           suggestions=suggestions,
                           replace=replacePlus)


if __name__ == '__main__':
    app.run()
