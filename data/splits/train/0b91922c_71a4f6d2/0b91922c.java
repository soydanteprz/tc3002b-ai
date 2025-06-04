import java.util.*;
import java.lang.*;
import java.io.*;   
import java.math.*;
public class Prac{     
    
    
    
    
    
    //static long lcm(long a,long b){return (a/gcd(a,b))*b;}
    static PrintWriter w = new PrintWriter(System.out);
    //static int r[]={0,1,0,-1}, c[]={1,0,-1,0};
    static int[] nextG(int arr[]){
        int n = arr.length;
        Stack<Integer> s = new Stack<>();
        int ng[] = new int[n];
        for(int i = 0 ; i < n ; i++){
            while(!s.isEmpty() && arr[s.peek()] <= arr[i]){
                ng[s.pop()] = i;
            }
            s.add(i);
        }
        while(!s.isEmpty()){
            ng[s.pop()] = n;
        }
        return ng;
    }
    static int[] nextS(int arr[]){
        int n = arr.length;
        Stack<Integer> s = new Stack<>();
        int ns[] = new int[n];
        for(int i = 0 ; i < n ; i++){
            while(!s.isEmpty() && arr[s.peek()] >= arr[i]){
                ns[s.pop()] = i;
            }
            s.add(i);
        }
        while(!s.isEmpty()){
            ns[s.pop()] = n;
        }
        return ns;
    }
    static int[] prevG(int arr[]){
        int n = arr.length;
        Stack<Integer> s = new Stack<>();
        int pg[] = new int[n];
        for(int i = n-1 ; i >= 0 ; i--){
            while(!s.isEmpty() && arr[s.peek()] <= arr[i]){
                pg[s.pop()] = i;
            }
            s.add(i);
        }
        while(!s.isEmpty()){
            pg[s.pop()] = -1;
        }
        return pg;
    }
    static int[] prevS(int arr[]){
        int n = arr.length;
        Stack<Integer> s = new Stack<>();
        int ps[] = new int[n];
        for(int i = n-1 ; i >= 0 ; i--){
            while(!s.isEmpty() && arr[s.peek()] >= arr[i]){
                ps[s.pop()] = i;
            }
            s.add(i);
        }
        while(!s.isEmpty()){
            ps[s.pop()] = -1;
        }
        return ps;
    }
    public static void main(String [] args){
        InputReader sc=new InputReader(System.in); 
        
        int n = sc.ni();
        int arr [] = sc.nia(n);
        int ng[] = nextG(arr);
        int ns [] = nextS(arr);
        int pg[] = prevG(arr);
        int ps[] = prevS(arr);
        int ans[]=new int[n];
        
        Arrays.fill(ans,10000000);
        ans[n-1] = 0;
        for(int i = n -1 ; i >= 0 ; i --){
            if(ns[i] != n){
                ans[i] = Math.min(ans[i] , ans[ns[i]]+1);
            }
            if(ng[i] != n){
                ans[i] = Math.min(ans[i] , ans[ng[i]]+1);
            }
            if(pg[i] != -1){
                ans[pg[i]] = Math.min(ans[pg[i]] , ans[i]+1);
            }
            if(ps[i] != -1){
                ans[ps[i]] = Math.min(ans[ps[i]] , ans[i]+1);
            }
        }
        
        w.println(ans[0]);
        w.close();
    }
}
