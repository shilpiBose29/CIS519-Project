# CIS519-Project
Airbnb Recommender System

## List of Features That Can Be Used

In `*_listings.csv`:

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

  ​