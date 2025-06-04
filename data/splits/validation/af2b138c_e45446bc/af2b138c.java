import java.util.*;
import java.lang.*;
import java.io.*;



public class ecf106 {

//HashMap<Integer,Integer> hm=new HashMap<>();
//ArrayList<Integer> list=new ArrayList<>();
//HashSet<Integer> hs=new HashSet<>();
//Stack<Integer> st=new Stack<>();
//Queue<Integer> q=new LinkedList<>();
//PriorityQueue<Integer> pq=new PriorityQueue<>();
// static ArrayList<ArrayList<Integer>> gp=new ArrayList<>();
//int[] arr=new int[n];
//System.out.println("YES");
//1 hr- atleast 2 qs d-2
//Mt_do 3 qs d-2  
//Mt_do 4 qs d-3
//StringBuilder sb=new StringBuilder();

public static void main(String[] args) throws Exception{
    InputStreamReader ip=new InputStreamReader(System.in);
    BufferedReader br = new BufferedReader(ip);
    int t= Integer.parseInt(br.readLine());
    
    while(t-->0){
       int n=Integer.parseInt(br.readLine());//Integer.parseInt(strs[0]);
        String[] strs=(br.readLine()).trim().split(" ");
        // int n=Integer.parseInt(strs[0]),k=Integer.parseInt(strs[1]);//,k=Integer.parseInt(strs[2]);

        // strs=(br.readLine()).trim().split(" ");
         long[] arr=new long [n];

        for(int i=0;i<n;i++){
            arr[i]=Integer.parseInt(strs[i]);
        }

        long min1=0,min2=0;
        long x=0,y=0;
        long hor=arr[0],vert=arr[1];
        long ans=(n*hor)+(n*vert);
        
        for(int i=2;i<n;i++){
            if(i%2==0){
                x++;
                if(arr[i]>=hor){   //v.v.imp  donot update horizontal
                    min1+=arr[i];  
                }else{
                    min1+=hor;  hor=arr[i];
                }
            }else{
                y++;
                if(arr[i]>=vert){   //v.v.imp  donot update vertical
                    min2+=arr[i];    
                }else{
                    min2+=vert;  vert=arr[i];
                }
            }

            long rem=(n-x)*hor+(n-y)*vert;

            ans=Math.min(ans, min1+min2+rem);
        }

        System.out.println(ans);
    }

    }
}
