import cv2
from flask import json, Blueprint

from app.core.target_detect.pointlocation import BAG_AND_LANDMARK, PointLocationService


main_blue = Blueprint('main_blue', __name__)

@main_blue.route('/hello', methods=['GET'])
def hello():
	return "ok"

@main_blue.route('/position', methods=['GET'])
def bag_position():
	im = cv2.imread('C:/work/imgs/test/bag8.bmp')
	with PointLocationService(img=im, print_or_no=False) as  a:
		a.location_objects(flag=BAG_AND_LANDMARK)
		positions=[(bag.x,bag.y) for bag in a.bags]
	# positions=[(1,2),(2,5)]
	return json.dumps(positions)