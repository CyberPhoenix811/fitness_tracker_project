import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/01_data_processed')
df.info()

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df['Set'] == 1]
plt.plot(set_df['acc_y'])

plt.plot(set_df['acc_y'].reset_index(drop = True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop = True), label = label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("Label == 'squat'").query("Participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(['Participant'])['acc_y'].plot()
ax.set_xlabel('samples')
ax.set_ylabel('accelerometer (Y axis)')
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("Label == 'squat'").sort_values("Participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(['Participant'])['acc_y'].plot()
ax.set_xlabel('samples')
ax.set_ylabel('accelerometer (Y axis)')
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = 'squat'
participant = 'A'
all_axis_df = df.query(f"Label == '{label}'").query(f"Participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax = ax)
ax.set_xlabel('samples')
ax.set_ylabel('accelerometer')
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

labels = df["Label"].unique()
participants = df["Participant"].sort_values().unique()
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"Label == '{label}'")
        .query(f"Participant == '{participant}'")
        .reset_index()
        )
        if len(all_axis_df) > 0:
        
            fig, ax = plt.subplots()
            all_axis_df[["acc_x","acc_y","acc_z"]].plot(ax = ax)
            ax.set_xlabel('samples')
            ax.set_ylabel('accelerometer')
            plt.title(f"{label}({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------
label = 'squat'
participant = 'A'
combined_df = (
    df.query(f"Label == '{label}'")
    .query(f"Participant == '{participant}'")
    .reset_index(drop = True)
)

fig, ax = plt.subplots(nrows= 2, sharex= True, figsize = (20,10))
combined_df[["acc_x","acc_y","acc_z"]].plot(ax = ax[0])
combined_df[["gyr_x","gyr_y","gyr_z"]].plot(ax = ax[1])

ax[0].legend(
    loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
)
ax[1].legend(
    loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
)
ax[1].set_xlabel('Samples')


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df["Label"].unique()
participants = df["Participant"].sort_values().unique()
for label in labels:
    for participant in participants:
        combined_df = (
            df.query(f"Label == '{label}'")
        .query(f"Participant == '{participant}'")
        .reset_index(drop = True)
        )
        if len(combined_df) > 0:
            fig, ax = plt.subplots(nrows= 2, sharex= True, figsize = (20,10))
            combined_df[["acc_x","acc_y","acc_z"]].plot(ax = ax[0])
            combined_df[["gyr_x","gyr_y","gyr_z"]].plot(ax = ax[1])

            ax[0].legend(
                loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
            )
            ax[1].legend(
                loc = "upper center", ncol = 3 , bbox_to_anchor = (0.5,1.15), fancybox = True, shadow = True
            )
            ax[1].set_xlabel('Samples')
            
            plt.savefig(f"../../reports/figures/{label.title()}({participant}).png")
            plt.show()