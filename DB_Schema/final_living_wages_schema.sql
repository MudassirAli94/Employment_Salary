CREATE SCHEMA living_wages_project;

CREATE TABLE living_wages_project.dim_dma ( 
	dma_id int64 NOT NULL  ,
	location_name string OPTIONS( description='name of location' )   
 );

ALTER TABLE living_wages_project.dim_dma ADD PRIMARY KEY ( dma_id )  NOT ENFORCED;

CREATE TABLE living_wages_project.dim_location ( 
	location_id string NOT NULL  ,
	state string OPTIONS( description='state in the USA' )   ,
	state_short string OPTIONS( description='two letter abbreviation for state' )   ,
	county string OPTIONS( description='county in the state' )   ,
	city string OPTIONS( description='city in the county' )   ,
	latitude numeric  ,
	longitude numeric  
 );

ALTER TABLE living_wages_project.dim_location ADD PRIMARY KEY ( location_id )  NOT ENFORCED;

CREATE TABLE living_wages_project.facts_jobs_salary ( 
	job_id string NOT NULL OPTIONS( description='primary key' )   ,
	company_name string  ,
	company_icon string  ,
	state string OPTIONS( description='state location of job' )   ,
	state_short string OPTIONS( description='two letter abbreviation of state location of job' )   ,
	city string  ,
	job_title string  ,
	job_family string  ,
	occupational_area string  ,
	salary int64  
 );

ALTER TABLE living_wages_project.facts_jobs_salary ADD PRIMARY KEY ( job_id )  NOT ENFORCED;

CREATE TABLE living_wages_project.facts_jobs ( 
	job_id string NOT NULL OPTIONS( description='foreign key for dim_employee table' )   ,
	dma_id int64 OPTIONS( description='foreign key to dma table' )   ,
	location_id string  ,
	mit_estimated_baseline_salary numeric  ,
	years_of_experience int64 OPTIONS( description='number of years of experience the person has for the job' )   ,
	years_at_level int64 OPTIONS( description='number of years the person has at the level' )   ,
	minimum_wage numeric OPTIONS( description='minimum wage per state' )   ,
	tipped_wage numeric  ,
	rank int64 OPTIONS( description='rank of DMA' )   ,
	tv_homes int64 OPTIONS( description='number of televisions per DMA data' )   ,
	percent_of_united_states numeric OPTIONS( description='DMA data' )   ,
	total_population int64 OPTIONS( description='total population of the county' )   ,
	total_population_density int64  ,
	total_land_area int64 OPTIONS( description='land area in sq miles' )   ,
	total_housing_units int64  ,
	total_occupied_housing_units int64  
 );

ALTER TABLE living_wages_project.facts_jobs ADD PRIMARY KEY ( job_id )  NOT ENFORCED;

ALTER TABLE living_wages_project.facts_jobs ADD CONSTRAINT job_id FOREIGN KEY ( job_id ) REFERENCES living_wages_project.facts_jobs_salary( job_id ) NOT ENFORCED;

ALTER TABLE living_wages_project.facts_jobs ADD CONSTRAINT dma_id FOREIGN KEY ( dma_id ) REFERENCES living_wages_project.dim_dma( dma_id ) NOT ENFORCED;

ALTER TABLE living_wages_project.facts_jobs ADD CONSTRAINT location_id FOREIGN KEY ( location_id ) REFERENCES living_wages_project.dim_location( location_id ) NOT ENFORCED;

