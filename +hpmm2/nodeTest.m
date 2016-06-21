classdef nodeTest < matlab.unittest.TestCase
    properties
        rng_seed
    end
    
    methods(TestMethodSetup)
        function random_save(testCase)
            % save current rng state
            testCase.rng_seed = rng();
            % set state to 1
            rng('default');
            rng(1);
        end
    end
    
    methods(TestMethodTeardown)
        function random_reset(testCase)
            % restore random number generator state
            rng(testCase.rng_seed);
        end
    end
    
    methods(Test)
        function test_calc_log_b_continuous(testCase)
            M = 3;
            K = 4;
            N = 100;
            
            id_child = (1:M)';
            X = rand(N,M);
            obj = hpmm2.node(id_child, K);
            for m = 1:M
                beta_children{m} = X(:,m);
            end
            obj.init_distribution(beta_children);
            
            for m = 1:M
                X_act = beta_children{m};
                [log_b] = calc_log_b(obj, X_act, m);
                log_b_exp = zeros(size(log_b));
                for k = 1:K
                    mu = obj.Mu{m}(k);
                    sigma = sqrt(obj.Sigma{m}(k));
                    log_b_exp(:,k) = log(normpdf(X_act, mu, sigma));
                end
                verifyEqual(testCase,log_b,log_b_exp,'RelTol',1e-13);
            end
            
        end
        function test_update_cpd_continuous_single_dim(testCase)
            M = 1;
            K = 4;
            N = 100;
            EM_steps = 10;

            X = rand(N,M);
            
            g = hpmm2.graph();
            g.init_data(X, ones(M,1));
            M_act = 1;
            id_node = g.add_node(M_act, K);
%             obj = hpmm2.node(id_child, K);
%             for m = 1:M
%                 beta_children{m} = X(:,m);
%             end
%             obj.init_distribution(beta_children);
            g.init_node_distribution(id_node);
            g.beta_pass(id_node);
            g.alpha_pass(id_node);
            
            % do 10 EM steps
            for i = 1:EM_steps
                g.update_prior(id_node);
                g.cpd_pass(id_node);
                g.beta_pass(id_node);
                g.alpha_pass(id_node);
            end
            
            
            [q, lklhd] = calc_q(g.nodes(id_node));
            lklhd = sum(lklhd);
            
            mu = g.nodes(id_node).Mu{M_act};
            sigma = shiftdim(g.nodes(id_node).Sigma{M_act},-2);
            p = g.nodes(id_node).alpha;
            gm = gmdistribution(mu,sigma,p);
            [q_exp, nlogl] = posterior(gm,X(:,M_act));
            lklhd_exp = -nlogl;
            
            verifyEqual(testCase,q,q_exp,'RelTol',1e-13);
            verifyEqual(testCase,lklhd,lklhd_exp,'RelTol',1e-13);
        end
        function test_update_cpd_continuous_multiple_dim(testCase)
            M = 5;
            K = 4;
            N = 100;
            EM_steps = 10;

            X = rand(N,M);
            
            g = hpmm2.graph();
            g.init_data(X, ones(M,1));
            id_child = (1:M)';
            id_node = g.add_node(id_child, K);
%             obj = hpmm2.node(id_child, K);
%             for m = 1:M
%                 beta_children{m} = X(:,m);
%             end
%             obj.init_distribution(beta_children);
            g.init_node_distribution(id_node);
            g.beta_pass(id_node);
            g.alpha_pass(id_node);
            
            % do 10 EM steps
            for i = 1:EM_steps
                g.update_prior(id_node);
                g.cpd_pass(id_node);
                g.beta_pass(id_node);
                g.alpha_pass(id_node);
            end
            
            
            [q, lklhd] = calc_q(g.nodes(id_node));
            lklhd = sum(lklhd);
            
            M_child = (1:length(id_child))';
            mu = cat(2,g.nodes(id_node).Mu{M_child});
            sigma = permute(cat(2,g.nodes(id_node).Sigma{M_child}),[3 2 1]);
            p = g.nodes(id_node).alpha;
            gm = gmdistribution(mu,sigma,p);
            [q_exp, nlogl] = posterior(gm,X(:,id_child));
            lklhd_exp = -nlogl;
            
            verifyEqual(testCase,q,q_exp,'RelTol',1e-13);
            verifyEqual(testCase,lklhd,lklhd_exp,'RelTol',1e-13);
        end
    end % methods (Test)
end % classedf

