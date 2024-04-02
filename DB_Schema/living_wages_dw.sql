CREATE SCHEMA cost_of_living_dw_lgl;

CREATE TABLE cost_of_living_dw_lgl.dm_company ( 
	company_id string NOT NULL  ,
	company_info_name string NOT NULL  ,
	company_info_icon string NOT NULL  ,
	num_employees int64 NOT NULL  ,
	date int64 NOT NULL  
 );

ALTER TABLE cost_of_living_dw_lgl.dm_company ADD PRIMARY KEY ( company_id )  NOT ENFORCED;

CREATE TABLE cost_of_living_dw_lgl.dm_employee ( 
	job_id string NOT NULL  ,
	job_family string NOT NULL  ,
	level string NOT NULL  ,
	occupational_area string NOT NULL  ,
	focus_tag string NOT NULL  ,
	years_of_experience string NOT NULL  ,
	years_at_company string NOT NULL  ,
	years_at_level string NOT NULL  ,
	work_arrangement string NOT NULL  ,
	employmentType string NOT NULL  ,
	company_id string NOT NULL  
 );

ALTER TABLE cost_of_living_dw_lgl.dm_employee ADD PRIMARY KEY ( job_id )  NOT ENFORCED;

CREATE TABLE cost_of_living_dw_lgl.dm_location ( 
	location_id string NOT NULL  ,
	state string NOT NULL  ,
	state_short string NOT NULL  ,
	county string NOT NULL  ,
	name string NOT NULL  ,
	county_or_city string NOT NULL  ,
	type_of_place string NOT NULL  ,
	geo_id string NOT NULL  ,
	ansi_code string NOT NULL  ,
	total_population int64 NOT NULL  
 );

ALTER TABLE cost_of_living_dw_lgl.dm_location ADD PRIMARY KEY ( location_id )  NOT ENFORCED;

CREATE TABLE cost_of_living_dw_lgl.dm_dma ( 
	dma_id int64 NOT NULL  ,
	rank int64 NOT NULL  ,
	tv_homes int64 NOT NULL  ,
	percent_of_united_states float NOT NULL  ,
	location_id string NOT NULL  
 );

ALTER TABLE cost_of_living_dw_lgl.dm_dma ADD PRIMARY KEY ( dma_id )  NOT ENFORCED;

CREATE TABLE cost_of_living_dw_lgl.facts_salary ( 
	fact_id string NOT NULL  ,
	salary int64 NOT NULL  ,
	minimum_wage float NOT NULL  ,
	tipped_wage float NOT NULL  ,
	dma_id int64 NOT NULL  ,
	job_id string NOT NULL  
 );

ALTER TABLE cost_of_living_dw_lgl.facts_salary ADD PRIMARY KEY ( fact_id )  NOT ENFORCED;

ALTER TABLE cost_of_living_dw_lgl.dm_dma ADD CONSTRAINT fk_dm_dma_dm_location FOREIGN KEY ( location_id ) REFERENCES cost_of_living_dw_lgl.dm_location( location_id ) NOT ENFORCED;

ALTER TABLE cost_of_living_dw_lgl.dm_employee ADD CONSTRAINT fk_dm_employee_dm_company FOREIGN KEY ( company_id ) REFERENCES cost_of_living_dw_lgl.dm_company( company_id ) NOT ENFORCED;

ALTER TABLE cost_of_living_dw_lgl.facts_salary ADD CONSTRAINT fk_facts_salary_dm_dma FOREIGN KEY ( dma_id ) REFERENCES cost_of_living_dw_lgl.dm_dma( dma_id ) NOT ENFORCED;

ALTER TABLE cost_of_living_dw_lgl.facts_salary ADD CONSTRAINT fk_facts_salary_dm_employee FOREIGN KEY ( job_id ) REFERENCES cost_of_living_dw_lgl.dm_employee( job_id ) NOT ENFORCED;
