
## Create and Modify Event Notifications in Data Masking

You can create and modify event notifications in Data Masking.

### Creating Event Notifications for Data Masking

In Data Safe you can create event notifications for Data Masking related
events. You can use the quickstart template for common events or the advanced
event notification workflows to create notifications.

Prerequisites:

Ensure you have the necessary IAM permissions to create event notifications.
For more information, see [Permissions to Use Contextual Event
Notifications](/pls/topic/lookup?ctx=en/cloud/paas/data-safe&id=ADMDS-
GUID-44143439-1BF8-4A36-B05F-4BEAF1741C7C) in the Administering Oracle Data
Safe guide.

To create notifications:

  1. Under Security center, click Data masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Click the Notifications tab. 
  4. Click Create notification. 

If you don't have any notifications created for the selected resource then you
will see a list of available quickstart templates. You may click on one of
these instead.

The Create notification side panel will appear.

  5. Select to create an event notification from either a Quickstart template or an Advanced event notification. 

A Quickstart templates allow you to select from a list of common event
scenarios. When you create a notification from a quickstart template, the Rule
and Event is created automatically.

Note:

The Rule and Event are created in the compartment that you were working in
when you started the Notification workflow. Rules and Events will only trigger
for the compartment and any child-compartments of the compartment that they
were created in.

  6. If you selected Quickstart in the previous step, make a quickstart Template selection.

If you selected Advanced event notification in the previous step, type in a
Rule name and select an Event type.

See [Data Masking Event Types](/pls/topic/lookup?ctx=en/cloud/paas/data-
safe&id=ADMDS-GUID-5D04B7FD-884E-47CE-BFAA-AC9A9A8E4FF5) in the Administering
Oracle Data Safe guide for more information on events.

  7. Select to either Create new topic or to Select existing topic. 
  8. Select a Compartment. 

Note:

This compartment is where the topic will be created, not where the rule and
event will be monitored in.

  9. If you're creating a new topic, type the topic name or, if you're using an existing topic, select the topic name.
  10. Select a Subscription protocol. 
  11. Provide the necessary inputs for the selected subscription protocol.
  12. Optionally, click Show Advanced Options to tag the notification. 
  13. Click Create notification. 

### Modifying Event Notifications For Data Masking

After creating event notifications in Data Masking in Oracle Data Safe, you
can modify the notifications you created.

To modify the event and rule:

  1. Under Security center, click Data masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Click the Notifications tab. 
  4. Click on an existing event from the Name column. 

Note:

You will only see the Events that were created directly within Data Safe.

This will bring you to the Rule details page which is part of Oracle Cloud
Infrastructure (OCI) Events Service. For more information, see the
[Events](https://docs.oracle.com/en-us/iaas/Content/Events/home.htm) section
of the OCI Documentation.

To modify the topic and subscription:

  1. Under Security center, click Data masking. 
  2. Under Related Resources, click Masking Policies. 
  3. Click the Notifications tab. 
  4. Click on an existing topic from the Topic column. 

Note:

You will only see the Topics that were created directly within Data Safe.

This will bring you to the Topic Details page which is part of Oracle Cloud
Infrastructure (OCI) Notifications. For more information, see the
[Notifications](https://docs.oracle.com/en-
us/iaas/Content/Notification/home.htm) section of the OCI Documentation.
