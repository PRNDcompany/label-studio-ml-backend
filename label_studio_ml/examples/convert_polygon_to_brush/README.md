# convert_polygon_to_brush

### Setting up the Label Studio server

#### Obtain your API token

Log into the Label Studio interface (in the example above, at
`http://<LABEL_STUDIO_HOST>:8080`). 

Go to the [**Account & Settings** 
page](https://labelstud.io/guide/user_account#Access-token), and make a note of the Access Token, which we will use later as
the `LABEL_STUDIO_ACCESS_TOKEN`.

### Setting up the Convert-Polygon-to-Brush (segmentation) backend

#### Clone the repository

Make a clone of this repository on your host system and move it into the working
directory.

```
git clone https://github.com/PRNDcompany/label-studio-ml-backend

cd label-studio-ml-backend
pip3 install -e .

cd label_studio_ml/examples/convert_polygon_to_brush
```

### Setting up the backend manually

#### Adjust variables and `_wsgi.py` depending on your choice of model

You can set the following environment variables to change the behavior of the model.

* `LABEL_STUDIO_HOST` sets the endpoint of the Label Studio host. Must begin with `http://` 
* `LABEL_STUDIO_ACCESS_TOKEN` sets the API access token for the Label Studio host.

#### Start the Backend

You can now manually start the ML backend.

```
python _wsgi.py
```

or

```bash
label-studio-ml start convert_polygon_to_brush/
```