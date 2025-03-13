import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.align.visual_extractor.parameters()))
    te_params = list(map(id, model.align.text_extractor.parameters()))
    tle_params = list(map(id, model.align.textlist_extractor.parameters()))


    other_params = filter(lambda x: id(x) not in ve_params + te_params + tle_params, model.parameters())

    optimizer_params = []

    optimizer_params.append({'params': filter(lambda p: p.requires_grad, model.align.visual_extractor.parameters()), 'lr': args.lr_ve})
    optimizer_params.append({'params': filter(lambda p: p.requires_grad, model.align.text_extractor.parameters()), 'lr': args.lr_te})
    optimizer_params.append({'params': filter(lambda p: p.requires_grad, model.align.textlist_extractor.parameters()), 'lr': args.lr_tle})


    optimizer_params.append({'params': filter(lambda p: p.requires_grad, other_params), 'lr': args.lr_other})

    optimizer = getattr(torch.optim, args.optim)(
        optimizer_params,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    return optimizer



def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
