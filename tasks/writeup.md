### What's wrong with a local-file audit log in a real production deployment? Name one concrete failure mode.
A local-file audit log has a durability problem. If the disk fails, the audit log can be permanently lost.

### If you were extending this CLI to production use, name one feature you'd add (other than policy enforcement — deliberately a non-feature here) and why.

I would add a dry run mode feature that shows what model and key metrics will be used without actually applying them.                                                                                                                                                                                                                     
This reduces the risk of accidentally promoting a wrong version to production.                                                                                                                                                                                                                                                                                                                    
  