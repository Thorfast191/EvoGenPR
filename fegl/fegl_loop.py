def fegl_cycle(gpr, senpr, dataloader, optimizer, ewc=None):
    senpr.train()
    for x, y in dataloader:
        preds = senpr(x)
        loss = focal_loss(preds, y)

        if ewc:
            loss += 0.4 * ewc.penalty(senpr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Identify weak samples
    with torch.no_grad():
        uncertainty = torch.sigmoid(preds).std(dim=1)

    hard_idx = uncertainty.topk(32).indices
    synthetic = gpr.generate_targeted(hard_idx)

    return synthetic
