import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Create and return a Spark session
    return: spark session object
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Staging, Extracting song data json files from AWS S3 and Writing Extracted parquet data file back to AWS S3
    
    Arguments:
            spark {object}:        The entry point to programming Spark with the Dataset and DataFrame API.
            input_data {string}:   S3 bucket where Sparkify's event data is stored 
            output_data {string}:  S3 bucket to store extracted parquet data file
    Returns:
            No return values
    
    """
    # get filepath to song data file
    song_data = input_data + "song_data/*/*/*/*.json"
    
    # read song data file
    df = spark.read.json(song_data, schema = get_song_schema())

    # extract columns to create songs table
    songs_table = df.select("song_id",
                            "title",
                            "artist_id",
                            "year",
                            "duration").dropDuplicates(["song_id"])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data + "songs_table.parquet",
                              partitionBy = ["year", "artist_id"],
                              mode = "overwrite") 

    # extract columns to create artists table
    artists_table = df.select("artist_id",
                              "artist_name",
                              "artist_location",
                              "artist_latitude",
                              "artist_longitude").dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + "artists_table.parquet",
                                mode = "overwrite")



def process_log_data(spark, input_data, output_data):
     """Staging, Extracting log data json files from AWS S3 and Writing Extracted parquet data file back to AWS S3
    
    Arguments:
            spark {object}:        The entry point to programming Spark with the Dataset and DataFrame API.
            input_data {string}:   S3 bucket path where Sparkify's event data is stored 
            output_data {string}:  S3 bucket path used to store extracted parquet data file
    Returns:
            No return values
    
    """
    # get filepath to log data file
    log_data =input_data + "log-data/*/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data, schema = get_log_schema())
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.selectExpr("userId as user_id",
                                "firstName as first_name",
                                "lastName as last_name",
                                "gender",
                                "level").dropDuplicates(["user_id"]) 

    
    # write users table to parquet files
    users_table.write.parquet(output_data + "users_table.parquet",
                              mode = "overwrite")


    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x / 1000)), Stamp())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp((x / 1000)), Stamp())
    df = df.withColumn("datetime", get_datetime(col("ts")))
    
    # extract columns to create time table
    time_table = df.selectExpr("timestamp as start_time",
                               "hour(timestamp) as hour",
                               "dayofmonth(timestamp) as day",
                               "weekofyear(timestamp) as week",
                               "month(timestamp) as month",
                               "year(timestamp) as year",
                               "dayofweek(timestamp) as weekday"
                               ).dropDuplicates(["start_time"])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time_table.parquet",
                             partitionBy = ["year", "month"],
                             mode = "overwrite")

    # read in song data to use for songplays table
    song_df = input_data + "song_data/*/*/*/*.json"
    song_df = spark.read.json(song_data, schema = get_song_schema())

    # extract columns from joined song and log datasets to create songplays table 
    song_df.createOrReplaceTempView("song_data")
    df.createOrReplaceTempView("log_data")
    songplays_table =  spark.sql("""
                                SELECT monotonically_increasing_id() as songplay_id,
                                ld.timestamp as start_time,
                                year(ld.timestamp) as year,
                                month(ld.timestamp) as month,
                                ld.userId as user_id,
                                ld.level as level,
                                sd.song_id as song_id,
                                sd.artist_id as artist_id,
                                ld.sessionId as session_id,
                                ld.location as location,
                                ld.userAgent as user_agent
                                FROM log_data ld
                                JOIN song_data sd
                                ON (ld.song = sd.title
                                AND ld.length = sd.duration
                                AND ld.artist = sd.artist_name)
                                """)


    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(output_data + "songplays_table.parquet",
                                  partitionBy=["year", "month"],
                                  mode="overwrite")


def main():
    """
    Extract songs and events data from S3, Transform it into dimensional tables format, and Load it back to S3 in Parquet format
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://udacity-de-sparkify-data-lake/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
