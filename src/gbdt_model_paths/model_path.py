


class ModelPath:

    def __init__(self, model, dataset):
        self.datset = dataset
        self.model = model
        self.model.fit(dataset.X_train, dataset.y_train)

        self.X_diff = dataset.df.loc[dataset.X_train.index, [dataset.target_col]]
        self.num_trees = len(model.get_booster().get_dump())
        for n in range(self.num_trees):
            self.X_diff[f'pred_{n}'] = model.predict(dataset.X_train, iteration_range=(0, n+1), output_margin=True)
            if n > 0:
                self.X_diff[f'diff_{n}'] = round(self.X_diff[f'pred_{n}'] - self.X_diff[f'pred_{n - 1}'], 5)
            else:
                self.X_diff[f'diff_{n}'] = round(self.X_diff[f'pred_{n}'], 5)

    def get_groups_info(self, grouped, print_stats=True):
        ratios = []
        counts = []
        for name, group in grouped:
            ratios.append(len(group[group[self.datset.target_col] == 1]) / len(group))
            counts.append(len(group))
        if print_stats:
            print(f'1: {len([r for r in ratios if r == 1])}, 0: {len([r for r in ratios if r == 0])}, all: {len(ratios)}')
            print(f'part of groups: {len([r for r in ratios if r == 1 or r == 0]) / len(ratios)}')
            print(f'part of examples: {sum([counts[i] for i in range(len(ratios)) if ratios[i] == 1 or ratios[i] == 0]) / sum(counts)}')
        return ratios, counts
