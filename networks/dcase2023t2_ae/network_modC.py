raise SystemExit('REMOVED: network_modC.py — experimental ModC network disabled')

    def __init__(self, input_dim, block_size, bottleneck_dim=32, n_layers=4):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.n_layers = n_layers
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # Build encoder dynamically
        encoder_layers = []
        # First layer: input_dim → 128
        encoder_layers.append(nn.Linear(self.input_dim, 128))
        encoder_layers.append(nn.BatchNorm1d(128, momentum=0.01, eps=1e-03))
        encoder_layers.append(nn.ReLU())
        
        # Hidden layers: 128 → 128 (n_layers - 1 times)
        for _ in range(n_layers - 1):
            encoder_layers.append(nn.Linear(128, 128))
            encoder_layers.append(nn.BatchNorm1d(128, momentum=0.01, eps=1e-03))
            encoder_layers.append(nn.ReLU())
        
        # Bottleneck layer: 128 → bottleneck_dim
        encoder_layers.append(nn.Linear(128, bottleneck_dim))
        encoder_layers.append(nn.BatchNorm1d(bottleneck_dim, momentum=0.01, eps=1e-03))
        encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder dynamically (symmetric)
        decoder_layers = []
        # First layer: bottleneck_dim → 128
        decoder_layers.append(nn.Linear(bottleneck_dim, 128))
        decoder_layers.append(nn.BatchNorm1d(128, momentum=0.01, eps=1e-03))
        decoder_layers.append(nn.ReLU())
        
        # Hidden layers: 128 → 128 (n_layers - 1 times)
        for _ in range(n_layers - 1):
            decoder_layers.append(nn.Linear(128, 128))
            decoder_layers.append(nn.BatchNorm1d(128, momentum=0.01, eps=1e-03))
            decoder_layers.append(nn.ReLU())
        
        # Output layer: 128 → input_dim (no activation)
        decoder_layers.append(nn.Linear(128, self.input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z), z
