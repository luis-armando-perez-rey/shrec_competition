# Script to create rendered images by rotating around the 3D scene. 
# To run the script open SketchUp command line and type 
# load /path_to_script/render_rotations

# Rendering parameters
num_views = 12 # number of views from scene obtained by encircling 360 degrees around the scene 12 is default
# resolution of images
x_resolution = 256 
y_resolution = 256

# Select directory with SKP models interactively 
dir = UI.select_directory(title: "Select directory of .skp models")
files = Dir.entries(dir).select{|e| e =~ /[.]skp$/ }
puts(files)

# Saving directory 
save_dir = UI.select_directory(title: "Select directory for saving renders")

# View parameters for rendering images
camera_view = Sketchup.active_model.active_view
phi = Math::PI/2
m = (0...num_views).to_a
pi_fraction = 2*Math::PI/num_views
views = m.map { |a| pi_fraction*a }

for file in files
	skp_file_path = dir+"/"+file
	# Important constants
	puts("Loading "+skp_file_path)
	
	
	# Load sketchup model
	path = dir+"/"+file
	Sketchup.open_file(path)


	ents = Sketchup.active_model.entities
	group=ents.add_group(Sketchup.active_model.entities.to_a)
	ctr = group.bounds.center
	group_bounds = group.local_bounds
	radius = group_bounds.diagonal
	group_height = group_bounds.height
	ctr.x = -ctr.x
	ctr.y = -ctr.y
	ctr.z = -ctr.z
	t = Geom::Transformation.translation ctr
	ents.transform_entities t, group
	puts("Loaded "+skp_file_path)

	

	for view in 0...num_views 
		puts("Rendering view "+String(view))
		theta = views[view]
		eye = [radius*(Math::cos(theta)*Math::sin(phi)), radius*Math::sin(theta)*Math::sin(phi), group_height]
		cam = Sketchup::Camera.new eye, [0,0,0], [0,0,1]
		camera_view.camera = cam
		filename = file.split(".skp")[0]
		Sketchup.active_model.active_view.write_image({
	        	:filename    => save_dir+"/"+filename+"_"+String(view)+"_.png",
        		:antialias   => true,
        		:transparent => true,
			:width => x_resolution,
			:height => y_resolution})
	end
	Sketchup.active_model.close(ignore_changes = true)
end