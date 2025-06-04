import java.util.*;
import java.io.*;
public class Main
{
	
    public static int ask(int i){
        FastReader sc = new FastReader();
        System.out.println("? " + (i+1));
        System.out.flush();
        int x = sc.nextInt();
        return x - 1;
    }
	public static void main(String[] args) {
		FastReader sc = new FastReader();
		int t = sc.nextInt();
		int inf = 1000000007;
		while(t-->0){
		    int n = sc.nextInt();
		    int ans[] = new int[n];
		    for(int i=0;i<n;i++){
		        if(ans[i] == 0){
		            ArrayList<Integer> cycle = new ArrayList<Integer>();
    		        int x = ask(i), y = ask(i);
    		        cycle.add(y);
    		        while(y != x){
    		            y = ask(i);
    		            cycle.add(y);
    		        }  
    		        for(int j=0;j<cycle.size();j++){
    		            ans[cycle.get(j)] = cycle.get((j+1)%cycle.size()) + 1;
    		        }
		        }
		    }
		    
		    System.out.print("! ");
		    for(int i=0;i<n;i++) System.out.print(ans[i] + " ");
		    System.out.println();
		}
	}
}
