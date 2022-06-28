import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    sns.set(style='whitegrid', font_scale=1.5)

    df = pd.read_csv('Non-Coop Agents - Sheet1-3.csv') # results stored in spread sheet
    df = df[df['Guess What Acc (all)'] > 0]

    GUESS_WHAT_ACC_ALL = 'Object ID Error (all)'
    GUESS_WHAT_ACC_COOP = 'Object ID Error (coop)'
    GUESS_WHAT_ACC_NONCOOP = 'Object ID Error (non-coop)'
    DETECTION_ACC = 'Coop ID Error'
    PERCENT_NON_COOP = 'Percent Cooperative'
    NON_COOP_ORACLE = 'Answer Player'
    REWARD = 'Strategy'
    LW = 3.5

    df[GUESS_WHAT_ACC_ALL] = 1 - df['Guess What Acc (all)']
    df[GUESS_WHAT_ACC_COOP] = 1 - df['Guess What Acc (coop)']
    df[GUESS_WHAT_ACC_NONCOOP] = 1 - df['Guess What Acc (non-coop)']
    df[DETECTION_ACC] = 1 - df['Detection Acc']
    df[PERCENT_NON_COOP] = 100 * (1 - df['Percent Non-Coop'])
    df[NON_COOP_ORACLE] = df['Non-Coop Oracle']
    df[REWARD] = df['Reward'].replace({
        'Guess What' : 'Obj. ID (12)',
        'Detection' : 'Coop. ID (11)',
        'None' : 'No RL',
        'Both' : 'Avg. of (11) & (12)',
        'Random' : 'Random Chance'
    })

    rewards = set(df[REWARD])

    LABELS = [
        (NON_COOP_ORACLE, 'original', PERCENT_NON_COOP, GUESS_WHAT_ACC_ALL, REWARD),
        (NON_COOP_ORACLE, 'history', PERCENT_NON_COOP, GUESS_WHAT_ACC_ALL, REWARD),
        (NON_COOP_ORACLE, 'image', PERCENT_NON_COOP, GUESS_WHAT_ACC_ALL, REWARD),
        (NON_COOP_ORACLE, 'both', PERCENT_NON_COOP, GUESS_WHAT_ACC_ALL, REWARD),
        (NON_COOP_ORACLE, 'original', PERCENT_NON_COOP, GUESS_WHAT_ACC_COOP, REWARD),
        (NON_COOP_ORACLE, 'history', PERCENT_NON_COOP, GUESS_WHAT_ACC_COOP, REWARD),
        (NON_COOP_ORACLE, 'image', PERCENT_NON_COOP, GUESS_WHAT_ACC_COOP, REWARD),
        (NON_COOP_ORACLE, 'both', PERCENT_NON_COOP, GUESS_WHAT_ACC_COOP, REWARD),
        (NON_COOP_ORACLE, 'original', PERCENT_NON_COOP, GUESS_WHAT_ACC_NONCOOP, REWARD),
        (NON_COOP_ORACLE, 'history', PERCENT_NON_COOP, GUESS_WHAT_ACC_NONCOOP, REWARD),
        (NON_COOP_ORACLE, 'image', PERCENT_NON_COOP, GUESS_WHAT_ACC_NONCOOP, REWARD),
        (NON_COOP_ORACLE, 'both', PERCENT_NON_COOP, GUESS_WHAT_ACC_NONCOOP, REWARD),
        (NON_COOP_ORACLE, 'original', PERCENT_NON_COOP, DETECTION_ACC, REWARD),
        (NON_COOP_ORACLE, 'history', PERCENT_NON_COOP, DETECTION_ACC, REWARD),
        (NON_COOP_ORACLE, 'image', PERCENT_NON_COOP, DETECTION_ACC, REWARD),
        (NON_COOP_ORACLE, 'both', PERCENT_NON_COOP, DETECTION_ACC, REWARD)
    ]

    for i, labels in enumerate(LABELS):
        if i % 4 == 0:
            if i > 0:
                plt.tight_layout()
                plt.savefig(f'z-non-coop-figures/acc-{i // 4}.pdf')
            fig, ax = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(20,5))
        split_col, split_val, x, y, hue = labels
        data = df[df[split_col] == split_val]
        sns.lineplot(data=data, x=x, y=y, hue=hue, ax=ax.flat[i % 4])
        ax.flat[i % 4].set_title(f'{split_col} = {split_val}')

    plt.tight_layout()
    plt.savefig(f'z-non-coop-figures/acc-{(i + 1) // 4}.pdf')

    sns.set(style='whitegrid', font_scale=1.8)

    # ORACLES ARE DIFFERENT
    for rew in rewards:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        data = df[df[REWARD] == rew]
        sns.lineplot(data=data, x=PERCENT_NON_COOP, y=GUESS_WHAT_ACC_NONCOOP,
            hue=NON_COOP_ORACLE, style=NON_COOP_ORACLE, linewidth=4)
        plt.xlim(50,90)
        plt.title('Effect of Different Non-Coop Answer-Players')
        leg = ax.legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(LW)
        plt.tight_layout()
        plt.savefig(f'z-non-coop-figures/oracle-effect-{rew}.pdf')

    sns.set(style='whitegrid', font_scale=1.5)

    # ONE ORACLE, DIFFERENT GUESS WHAT METRICS
    for oracle in ['original', 'history', 'image', 'both']:
        fig, ax = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(16,4.1))
        data = df[df[NON_COOP_ORACLE]==oracle]
        labels = [GUESS_WHAT_ACC_ALL, GUESS_WHAT_ACC_COOP, GUESS_WHAT_ACC_NONCOOP, DETECTION_ACC]
        for i, ylabel in enumerate(labels):
            sns.lineplot(data=data, x=PERCENT_NON_COOP, y=ylabel,
                hue=REWARD, style=REWARD, linewidth=4, ax=ax.flat[i])
            ax.flat[i].set_ylabel('Error Rate')
            ax.flat[i].set_title(ylabel)
            if i != 2:
                ax.flat[i].get_legend().remove()
            else:
                leg = ax.flat[i].legend()
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(LW)
        plt.tight_layout()
        plt.savefig(f'z-non-coop-figures/performance-oracle={oracle}.pdf')

    sns.set(style='whitegrid', font_scale=1.8)
    
    # ONE ORACLE, DETECTION METRIC DIFFERENT REWARDS
    for oracle in ['original', 'history', 'image', 'both']:
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        data = df[df[NON_COOP_ORACLE]==oracle]
        sns.lineplot(data=data, x=PERCENT_NON_COOP, y=DETECTION_ACC,
            hue=REWARD, style=REWARD, linewidth=4)
        plt.title('Detection Error')
        plt.ylabel('Error')
        leg = ax.legend()

        for legobj in leg.legendHandles:
            legobj.set_linewidth(LW)

        if oracle == 'image' or oracle == 'both':
            # Get the bounding box of the original legend
            bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

            # Change to location of the legend. 
            xOffset = .1
            bb.x0 += xOffset
            bb.x1 += xOffset
            yOffset = .3
            bb.y0 += yOffset
            bb.y1 += yOffset
            leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

        elif oracle == 'history':
            # Get the bounding box of the original legend
            bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

            # Change to location of the legend. 
            xOffset = .08
            bb.x0 += xOffset
            bb.x1 += xOffset
            yOffset = .1
            bb.y0 += yOffset
            bb.y1 += yOffset
            leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

        elif oracle == 'original':
            # Get the bounding box of the original legend
            bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

            # Change to location of the legend. 
            xOffset = .08
            bb.x0 += xOffset
            bb.x1 += xOffset
            yOffset = -.08
            bb.y0 += yOffset
            bb.y1 += yOffset
            leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

        plt.tight_layout()
        plt.savefig(f'z-non-coop-figures/d-performance-oracle={oracle}.pdf')
        