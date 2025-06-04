import java.util.*;



import java.io.*;

public class code  {
	// remember long, to reformat ctrl + shift +f
	static Scanner sc = new Scanner(System.in);
	static PrintWriter pw = new PrintWriter(System.out);
 
	  
 
	public static void main(String[] args) throws Exception {

		int t=sc.nextInt();
		while(t-->0) {
            int n=sc.nextInt();
            int []vals=new int[n];
            boolean numLine[]=new boolean[n+1];

            for(int i=0;i<n;i++)vals[i]=sc.nextInt();
            String s=sc.nextLine();
            ArrayList<Integer>b=new ArrayList<Integer>();
            ArrayList<Integer>r=new ArrayList<Integer>();
            for(int i=0;i<n;i++) {
            	if(s.charAt(i)=='B' && vals[i]>0 )b.add(vals[i]);
            	else if( s.charAt(i)=='R' && vals[i]<=n)r.add(vals[i]);
            }
          Collections.sort(b);
          Collections.sort(r);
          int small=1;
          for(int i=0;i<b.size();i++) {
        	  
        	  int y=b.get(i);
        	  if(y<small)continue;
        	  numLine[small]=true;
        	  small++;
          }
         // pw.println(Arrays.toString(numLine));
          int large=n;
          for(int i=r.size()-1;i>=0;i--) {
        	  
        	  int y=r.get(i);
        	  if(y>large)continue;
        	 // y=Math.max(large, y);
        	  numLine[large]=true;
        	  large--;
          }
          //pw.print(Arrays.toString(numLine));
          boolean can=true;
          for(int i=1;i<=n;i++) {
        	  if(numLine[i]==false) {
        		  pw.println("no");
        		  can=false;
        		  break;
        	  }
        	  
          }
          if(can)pw.println("yes");
          
           
            	
            	
            

			
		}
		
	 pw.close();

	}

	// --------------------stuff ----------------------
	

	

}
