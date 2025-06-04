import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.abs;

import java.util.*;
import java.io.*;
import java.math.*;


public class C{

    static int mod = (int) (Math.pow(10, 9)+7);
	static final int dx[] = { -1, 0, 1, 0 }, dy[] = { 0, -1, 0, 1 };
	static final int[] dx8 = { -1, -1, -1, 0, 0, 1, 1, 1 }, dy8 = { -1, 0, 1, -1, 1, -1, 0, 1 };
	static final int[] dx9 = { -1, -1, -1, 0, 0, 0, 1, 1, 1 }, dy9 = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };

	static final double eps = 1e-10;
	static List<Integer> primeNumbers = new ArrayList<>();
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        //code below
        int test = sc.nextInt();
        while(test --> 0){
            int n = sc.nextInt();
            int[][] data = new int[n-1][2];

            Map<Integer, Integer> count = new HashMap<>();
            Graph g = new Graph(n+1);

            for(int i= 0; i < n-1; i++){
                int x = sc.nextInt();
                int y = sc.nextInt();

                data[i][0] = x;
                data[i][1] = y;

                g.addEdge(x, y);
            
                count.putIfAbsent(x, 0);
                count.put(x, count.get(x)+1);

                count.putIfAbsent(y, 0);
                count.put(y, count.get(y)+1);
            }

            //we will count 
            int one = 0;
            int two = 0;

            boolean flag = true;
            int start = 0;
            for(int i : count.keySet()){
                if(count.get(i) == 1){  
                    start = i;
                    one++;
                }else if(count.get(i) == 2){
                    two++;
                }else{
                    flag = false;
                    break;
                }
            }

            if(one != 2){
                flag = false;
            }

            if(!flag){
                out.println(-1);
                continue;
            }

            //now we know this question is solvable for sure
            //we do something really cool
            Map<Integer, ArrayList<Pair>> fuck = new HashMap<>();
            g.addData(start, fuck);
            
            for(int i = 0; i < n-1; i++){
                int x = data[i][0];
                int y = data[i][1];

                //search for the answer 
                ArrayList<Pair> temp = fuck.get(x);
                for(Pair curr : temp){
                    if(curr.a == y){
                        out.print(curr.b + " ");
                        break;
                    }
                }
            }
            
            out.println();
        }


        out.close();
    }

    //Updation Required
    //Fenwick Tree (customisable)
    //Segment Tree (customisable)

    //-----CURRENTLY PRESENT-------//
    //Graph
    //DSU
    //powerMODe
    //power
    //Segment Tree (work on this one) 
    //Prime Sieve
    //Count Divisors
    //Next Permutation 
    //Get NCR 
    //isVowel
    //Sort (int)
    //Sort (long)
    //Binomial Coefficient
    //Pair
    //Triplet
    //lcm (int & long)
    //gcd (int & long)
    //gcd (for binomial coefficient)
    //swap (int & char)
    //reverse

    //Fast input and output 

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    //-------------------------------------------------------------------

    //GRAPH (basic structure)
    public static class Graph{
        public int V;
        public ArrayList<ArrayList<Integer>> edges;

        //2 -> [0,1,2] (current)
        Graph(int V){
            this.V  = V;
            edges = new ArrayList<>(V+1);
            for(int i= 0; i <= V; i++){
                edges.add(new ArrayList<>());
            }
        }

        public void addEdge(int from , int to){
            edges.get(from).add(to);
            edges.get(to).add(from);
        }

        public void addData(int start, Map<Integer, ArrayList<Pair>> fuck){
            //we will add stuff here
            int curr = start;
            int[] visited = new int[V+1];
            visited[curr] = 1;

            int index = 0;
            while(true){
                int make = 0;
                boolean last = true;
                for(int edge : edges.get(curr)){
                    if(visited[edge] == 1){
                        continue;
                    }else{
                        make = edge; 
                        last = false;
                        break;
                    }
                }

                if(last){
                    break;
                }

                //now we have found make, we make and add it in map
                fuck.putIfAbsent(curr, new ArrayList<>());
                fuck.putIfAbsent(make, new ArrayList<>());

                fuck.get(curr).add(new Pair(make, index %2 == 0 ? 2 : 3));
                fuck.get(make).add(new Pair(curr, index %2 == 0 ? 2 : 3));

                visited[make] = 1;
                curr = make;
                index++;
            }
        }
    }

    //DSU (path and rank optimised)
    

    //with mod
    

    //without mod
    

    

    //prime sieve
    

    //number of divisors
    

    //now adding next permutation function to java hehe
    

    //finding the value of NCR in O(RlogN) time and O(1) space
    

    //is vowel function 
       

    //to sort the array with better method
	

    //sort long
    
	
    //for calculating binomialCoeff
    

    //Pair with int int 
    public static class Pair{
        public int a;
        public int b;

        Pair(int a , int b){
            this.a = a;
            this.b = b;
        }

        
    }

    //Triplet with int int int
    

    //Shortcut function
    

    //let's make one for calculating lcm basically
    

    //int version for gcd
    

    //long version for gcd
    

     //for ncr calculator(ignore this code)
     

    //swapping two elements in an array
    

    //for char array
    

    //reversing an array
    


    

    //SOME EXTRA DOPE FUNCTIONS 
    

    

    
   
    

    

    //O(n) every single time remember that
    

    // ll *fact = new ll[2 * n + 1];
	// ll *ifact = new ll[2 * n + 1];
	// fact[0] = 1;
	// ifact[0] = 1;
	// for (ll i = 1; i <= 2 * n; i++)
	// {
	// 	fact[i] = mod_mul(fact[i - 1], i, MOD1);
	// 	ifact[i] = mminvprime(fact[i], MOD1);
	// }
    //ifact is basically inverse factorial in here!!!!!(imp)
    

  

     
    //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;
      
    //-----------MyScanner class for faster input----------
    
   //--------------------------------------------------------
}
