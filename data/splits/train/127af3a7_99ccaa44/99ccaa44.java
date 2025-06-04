import java.util.*;
 
public class Main
{
	public static void main(String[] args) {
	    Scanner sc = new Scanner(System.in); 
		int tc = sc.nextInt();
		while(tc>0)
		{
		    tc--;
		    int ans = 0;
		    int n = sc.nextInt();
		    int in[] = new int[n];
		    int wx[] = new int[n-1];
		    int wy[] = new int[n-1];
		    ArrayList<ArrayList<Integer>> arr = new ArrayList<ArrayList<Integer>>();
		    HashMap<String,Integer> h = new HashMap<String,Integer>();
		    HashSet<Integer> h2 = new HashSet<Integer>();
		    for(int i=0;i<n;i++)
		    {
		        arr.add(new ArrayList<Integer>());
		    }
		    for(int i=0;i<n-1;i++)
		    {
		        int x = sc.nextInt();
		        int y = sc.nextInt();
		        x--;
		        y--;
		        in[x]++;
		        in[y]++;
		        if(in[x]>2 || in[y]>2)
		        {
		            ans = -1;
		        }
		        else if(ans!=-1)
		        {
		            arr.get(x).add(y);
		            arr.get(y).add(x);
		            wx[i] = x;
		            wy[i] = y;
		        }
		        
		    }
		    if(ans == -1)
		    {
		        System.out.println("-1");
		    }
		    else
		    {
		        int vis[] = new int[n];
		        ArrayDeque<Integer> q = new ArrayDeque<Integer>();
		        q.add(0);
		        vis[0] = 1;
		        while(q.size()>0)
		        {
		            int x = q.removeFirst();
		            for(int i=0;i<arr.get(x).size();i++)
		            {
		                int y = arr.get(x).get(i);
		                String temp  =String.valueOf(x);
		                if(vis[y] == 0)
		                {
		                    vis[y] = 1;
		                    if(h2.contains(y) || h2.contains(x))
		                    {
		                        String s1 = temp+"_"+String.valueOf(y);
		                        String s2 = String.valueOf(y)+"_"+temp;
		                        h.put(s1,11);
		                        h.put(s2,11);
		                        
		                    }
		                    else
		                    {
		                        String s1 = temp+"_"+String.valueOf(y);
		                        String s2 = String.valueOf(y)+"_"+temp;
		                        h.put(s1,2);
		                        h.put(s2,2);
		                        h2.add(x);
		                        h2.add(y);
		                    }
		                    q.add(y);
		                }
		            }
		        }
		        for(int i=0;i<wx.length;i++)
		        {
		            String s1 = String.valueOf(wx[i])+"_"+String.valueOf(wy[i]);
		            System.out.print(h.get(s1)+" ");
		        }
		        System.out.println();
		    }
		}
	}
}
