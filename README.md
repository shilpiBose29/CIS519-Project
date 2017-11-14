# CIS519-Project
Airbnb Recommender System

## List of Features That Can Be Used

### In `*_listings.csv`

- **Amenities**: Currently, there are 42 columns starting with `AMN_`, ending with:

  > 24-hour check-in, air conditioning, breakfast, buzzer/wireless intercom, cable tv, carbon monoxide detector, cat(s), dog(s), doorman, dryer, elevator in building, essentials, family/kid friendly, fire extinguisher, first aid kit, free parking on premises, gym, hair dryer, hangers, heating, hot tub, indoor fireplace, internet, iron, kitchen, laptop friendly workspace, lock on bedroom door, other pet(s), pets allowed, pets live on this property, pool, safety card, shampoo, smoke detector, smoking allowed, suitable for events, translation missing: en.hosting_amenity_49, translation missing: en.hosting_amenity_50, tv, washer, wheelchair accessible and wireless internet.

  All data are 0/1 values.

- **Cancellation Policy**: The `cancellation_policy` column. Possible values: `1,2,3,4`, with mapping:

  | Meaning         | Value |
  | --------------- | ----- |
  | flexible        | 1     |
  | moderate        | 2     |
  | strict          | 3     |
  | super_strict_30 | 4     |

- **Room Type**: The `room_type` column. Possible values: `1,2,3`, with mapping:

  | Meaning         | Value |
  | --------------- | ----- |
  | Shared room     | 1     |
  | Private room    | 2     |
  | Entire home/apt | 3     |

- **Property Type**: Expanded `property_type` into` ['POPTY=Apartment', 'POPTY=Bed & Breakfast', 'POPTY=Bungalow', 'POPTY=Cabin', 'POPTY=Camper/RV', 'POPTY=Chalet', 'POPTY=Condominium', 'POPTY=Dorm', 'POPTY=Earth House', 'POPTY=House', 'POPTY=Loft', 'POPTY=Other', 'POPTY=Tent', 'POPTY=Townhouse', 'POPTY=Treehouse', 'POPTY=Villa', 'POPTY=Yurt']` .

- **Bed Type**: The `bed_type` column. Possible values: `1,2,3,4,5`, with mapping:

  | Meaning         | Value |
  | --------------- | ----- |
  | Shared room     | 1     |
  | Private room    | 2     |
  | Entire home/apt | 3     |

- Expanded `bed_type` into `['BED=Airbed', 'BED=Couch', 'BED=Futon', 'BED=Pull-out Sofa', 'BED=Real Bed']` .

## List of Features Dropped

### In `*_listings.csv`

These columns are dropped:

```python
['calculated_host_listings_count', 'calendar_last_scraped', 'calendar_updated', 'city', 'country', 'country_code', 'description', 'has_availability', 'host_about', 'host_has_profile_pic', 'host_id', 'host_location', 'host_name', 'host_neighbourhood', 'host_picture_url', 'host_response_time', 'host_thumbnail_url', 'host_url', 'jurisdiction_names', 'last_scraped', 'last_searched', 'license', 'listing_url', 'market', 'medium_url', 'name', 'neighborhood_overview', 'neighbourhood_cleansed', 'notes', 'picture_url', 'region_id', 'region_name', 'region_parent_id', 'scrape_id', 'smart_location', 'space', 'state', 'street', 'summary', 'thumbnail_url', 'transit', 'xl_picture_url', 'zipcode']
```

